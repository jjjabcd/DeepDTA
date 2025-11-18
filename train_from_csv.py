import os

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import sys
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

tf.config.optimizer.set_jit(False)

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

def set_gpu(gpu_id):
    if gpu_id is None:
        print("[Info] GPU not specified, using default visible devices.")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[Warn] No GPU found. Using CPU.")
        return

    try:
        tf.config.set_visible_devices(gpus[int(gpu_id)], "GPU")
        tf.config.experimental.set_memory_growth(gpus[int(gpu_id)], True)
        print(f"[Info] Using GPU: {gpu_id}")
    except Exception as e:
        print("[Warn] GPU setting failed -> Using CPU.", e)

# -------------------------------------------------------------------------
# Tokenizers
# -------------------------------------------------------------------------
SMILES_VOCAB = list("#%)(+-.0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ@[]\\abdefghilmnoprstuy")
FASTA_VOCAB  = list("ACDEFGHIKLMNPQRSTVWY")

def build_index(vocab):
    return {ch: idx + 1 for idx, ch in enumerate(vocab)}

SMILES_INDEX = build_index(SMILES_VOCAB)
FASTA_INDEX  = build_index(FASTA_VOCAB)

def encode_sequence(seq, index, max_len):
    ids = [index.get(ch, 0) for ch in str(seq)[:max_len]]
    ids += [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)

# -------------------------------------------------------------------------
# Load CSV
# -------------------------------------------------------------------------
def load_csv(path, task_name="Kd", smiles_max_len=100, fasta_max_len=1000):
    df = pd.read_csv(path)

    label_col = "pKd" if task_name == "Kd" else "pKi"
    if label_col not in df.columns:
        raise ValueError(f"[Error] Label column '{label_col}' not found in {path}")

    Xs = np.stack([encode_sequence(s, SMILES_INDEX, smiles_max_len) for s in df.SMILES])
    Xf = np.stack([encode_sequence(s, FASTA_INDEX,  fasta_max_len) for s in df.FASTA])
    y  = df[label_col].values.astype(np.float32)

    return Xs, Xf, y

# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------
def build_model(smiles_len, fasta_len):
    kernel_init = initializers.GlorotUniform(seed=SEED)

    # SMILES
    smi_input = layers.Input((smiles_len,), name="smiles")
    x = layers.Embedding(len(SMILES_INDEX)+1, 128)(smi_input)
    x = layers.Conv1D(32, 4, activation="relu")(x)
    x = layers.Conv1D(64, 4, activation="relu")(x)
    x = layers.Conv1D(96, 4, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)

    # FASTA
    fa_input = layers.Input((fasta_len,), name="fasta")
    y = layers.Embedding(len(FASTA_INDEX)+1, 128)(fa_input)
    y = layers.Conv1D(32, 8, activation="relu")(y)
    y = layers.Conv1D(64, 8, activation="relu")(y)
    y = layers.Conv1D(96, 8, activation="relu")(y)
    y = layers.GlobalMaxPooling1D()(y)

    z = layers.concatenate([x, y])
    z = layers.Dense(1024, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(1024, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(512, activation="relu")(z)
    out = layers.Dense(1)(z)

    model = models.Model(inputs=[smi_input, fa_input], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
    return model

# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------
def train(args):

    set_seed(SEED)
    set_gpu(args.gpu)

    print(f"[Info] Loading Train data from {args.train_csv}...")
    Xs_tr, Xf_tr, y_tr = load_csv(args.train_csv, args.task_name,
                                  args.smiles_max_len, args.fasta_max_len)

    print(f"[Info] Loading Validation data from {args.val_csv}...")
    Xs_va, Xf_va, y_va = load_csv(args.val_csv, args.task_name,
                                  args.smiles_max_len, args.fasta_max_len)

    print(f"[Info] Loading Test data from {args.test_csv}...")
    Xs_te, Xf_te, y_te = load_csv(args.test_csv, args.task_name,
                                  args.smiles_max_len, args.fasta_max_len)

    print(f"[Info] Train size: {len(y_tr)}, Validation size: {len(y_va)}, Test size: {len(y_te)}")

    os.makedirs(args.out_dir, exist_ok=True)

    model = build_model(args.smiles_max_len, args.fasta_max_len)

    ckpt_path = os.path.join(args.out_dir, "best.h5")
    ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True,
                                              monitor="val_loss", mode="min",
                                              save_weights_only=True)

    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                             patience=args.patience,
                                             restore_best_weights=True)

    print("[Info] Starting training...")
    history = model.fit(
        {"smiles": Xs_tr, "fasta": Xf_tr},
        y_tr,
        validation_data=({"smiles": Xs_va, "fasta": Xf_va}, y_va),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[ckpt, early],
        verbose=1
    )

    # Save history
    pd.DataFrame(history.history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)

    # Load best weights
    model.load_weights(ckpt_path)
    preds = model.predict({"smiles": Xs_te, "fasta": Xf_te}, verbose=0).reshape(-1)

    mae = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(np.mean((y_te - preds)**2))
    pcc = pearsonr(y_te, preds)[0]

    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nPCC: {pcc:.4f}\n")

    print(f"[Info] Evaluation → MAE={mae:.4f}, RMSE={rmse:.4f}, PCC={pcc:.4f}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", type=str, required=True)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--smiles_max_len", type=int, default=100)
    p.add_argument("--fasta_max_len", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--gpu", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)

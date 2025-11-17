import os
import sys
import warnings
import argparse
import random
import numpy as np
import pandas as pd

# -------------------------
# Matplotlib (Non-interactive backend)
# -------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# Logging / Warnings Off
# -------------------------
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# -------------------------
# TensorFlow
# -------------------------
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# -------------------------
# GLOBAL SEED
# -------------------------
SEED = 42

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = "0"

    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# -------------------------
# Tokenizers
# -------------------------
SMILES_VOCAB = list("#%)(+-.0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ@[]\\abdefghilmnoprstuy")
FASTA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")

def build_char_index(vocab):
    return {ch: i + 1 for i, ch in enumerate(vocab)}

SMILES_INDEX = build_char_index(SMILES_VOCAB)
FASTA_INDEX = build_char_index(FASTA_VOCAB)

def encode_sequence(seq, index, max_len):
    seq = str(seq)
    ids = [index.get(ch, 0) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)


# -------------------------
# CSV Loader
# -------------------------
def load_csv(path, task_name="Kd", smiles_col="SMILES", fasta_col="FASTA",
             smiles_max_len=100, fasta_max_len=1000):

    if not os.path.exists(path):
        raise FileNotFoundError(f"[Error] File not found: {path}")

    df = pd.read_csv(path)

    # Select label column automatically
    if task_name == "Kd":
        label_col = "pKd"
    elif task_name == "Ki":
        label_col = "pKi"
    else:
        raise ValueError(f"[Error] Invalid task_name={task_name}. Must be one of ['Kd','Ki'].")

    for col in [smiles_col, fasta_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"[Error] Column '{col}' missing in CSV {path}. Columns = {df.columns.tolist()}")

    X_smiles = np.stack([encode_sequence(s, SMILES_INDEX, smiles_max_len) for s in df[smiles_col]])
    X_fasta  = np.stack([encode_sequence(s, FASTA_INDEX,  fasta_max_len) for s in df[fasta_col]])
    y        = df[label_col].astype(np.float32).values

    return X_smiles, X_fasta, y


# -------------------------
# DeepDTA-like Model
# -------------------------
def build_deepdta_like_model(smiles_len=100, fasta_len=1000,
                             smiles_vocab_size=len(SMILES_INDEX) + 1,
                             fasta_vocab_size=len(FASTA_INDEX) + 1,
                             emb_dim=128):

    kernel_init = initializers.GlorotUniform(seed=SEED)
    bias_init = initializers.Zeros()

    # SMILES
    inp_smi = layers.Input(shape=(smiles_len,), name="smiles")
    x = layers.Embedding(smiles_vocab_size, emb_dim, mask_zero=False,
                         embeddings_initializer=initializers.RandomUniform(seed=SEED))(inp_smi)
    x = layers.Conv1D(32, 4, activation="relu", kernel_initializer=kernel_init)(x)
    x = layers.Conv1D(64, 4, activation="relu", kernel_initializer=kernel_init)(x)
    x = layers.Conv1D(96, 4, activation="relu", kernel_initializer=kernel_init)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # FASTA
    inp_fa = layers.Input(shape=(fasta_len,), name="fasta")
    y = layers.Embedding(fasta_vocab_size, emb_dim, mask_zero=False,
                         embeddings_initializer=initializers.RandomUniform(seed=SEED))(inp_fa)
    y = layers.Conv1D(32, 8, activation="relu", kernel_initializer=kernel_init)(y)
    y = layers.Conv1D(64, 8, activation="relu", kernel_initializer=kernel_init)(y)
    y = layers.Conv1D(96, 8, activation="relu", kernel_initializer=kernel_init)(y)
    y = layers.GlobalMaxPooling1D()(y)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(1024, activation="relu")(z)
    z = layers.Dropout(0.2, seed=SEED)(z)
    z = layers.Dense(1024, activation="relu")(z)
    z = layers.Dropout(0.2, seed=SEED)(z)
    z = layers.Dense(512, activation="relu")(z)

    out = layers.Dense(1)(z)

    model = models.Model(inputs=[inp_smi, inp_fa], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


# -------------------------
# Plot utility
# -------------------------
def plot_history(history, out_dir):
    if "loss" not in history.history:
        return

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=300)
    plt.close()


# -------------------------
# TRAINING
# -------------------------
def train(args):

    # Set global seeds
    set_global_seed(SEED)

    # -------------------------
    # GPU setup
    # -------------------------
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[INFO] Using GPU: {args.gpu}")

    if tf.config.list_physical_devices("GPU"):
        try:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    # -------------------------
    # Load datasets
    # -------------------------
    print(f"[INFO] Loading train dataset: {args.train_csv}")
    Xs_tr, Xf_tr, y_tr = load_csv(args.train_csv, task_name=args.task_name,
                                  smiles_max_len=args.smiles_max_len,
                                  fasta_max_len=args.fasta_max_len)

    print(f"[INFO] Loading validation dataset: {args.val_csv}")
    Xs_va, Xf_va, y_va = load_csv(args.val_csv, task_name=args.task_name,
                                  smiles_max_len=args.smiles_max_len,
                                  fasta_max_len=args.fasta_max_len)

    print(f"[INFO] Loading test dataset: {args.test_csv}")
    Xs_te, Xf_te, y_te = load_csv(args.test_csv, task_name=args.task_name,
                                  smiles_max_len=args.smiles_max_len,
                                  fasta_max_len=args.fasta_max_len)

    print(f"[INFO] Dataset sizes → Train={len(y_tr)}, Val={len(y_va)}, Test={len(y_te)}")

    # -------------------------
    # Build model
    # -------------------------
    model = build_deepdta_like_model(smiles_len=args.smiles_max_len,
                                     fasta_len=args.fasta_max_len)

    # -------------------------
    # Callbacks
    # -------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_path = os.path.join(args.out_dir, "best.h5")
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(args.out_dir, "training_log.csv")
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
        verbose=1
    )

    # -------------------------
    # Training
    # -------------------------
    print("[INFO] Starting training...")
    history = model.fit(
        x={"smiles": Xs_tr, "fasta": Xf_tr},
        y=y_tr,
        validation_data=({"smiles": Xs_va, "fasta": Xf_va}, y_va),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[ckpt, early_stop, csv_logger],
        verbose=1
    )

    # Save history
    pd.DataFrame(history.history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)
    plot_history(history, args.out_dir)

    # -------------------------
    # Evaluation
    # -------------------------
    print("[INFO] Loading best weights...")
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    preds = model.predict({"smiles": Xs_te, "fasta": Xf_te}, verbose=0).reshape(-1)

    mae = mean_absolute_error(y_te, preds)
    rmse = float(np.sqrt(np.mean((y_te - preds)**2)))
    pcc = float(pearsonr(y_te, preds)[0]) if len(y_te) > 1 else float("nan")

    metrics_path = os.path.join(args.out_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nPCC: {pcc:.4f}\n")

    print(f"[RESULT] MAE={mae:.4f}, RMSE={rmse:.4f}, PCC={pcc:.4f}")
    print(f"[INFO] Metrics saved at {metrics_path}")


# -------------------------
# Argument Parser
# -------------------------
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", type=str, required=True, choices=["Kd", "Ki"])
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--smiles_max_len", type=int, default=100)
    parser.add_argument("--fasta_max_len", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./results")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)

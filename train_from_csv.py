import os
import sys
import warnings
import argparse
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------ Global seed & logging setup ------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

tf.random.set_seed(SEED)
tf.compat.v1.set_random_seed(SEED)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

if hasattr(sys, '_getframe'):
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

# ------------------ Tokenizers ------------------
SMILES_VOCAB = list("#%)(+-.0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ@[]\\abdefghilmnoprstuy")
FASTA_VOCAB  = list("ACDEFGHIKLMNPQRSTVWY")

def build_char_index(vocab):
    return {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 = PAD

SMILES_INDEX = build_char_index(SMILES_VOCAB)
FASTA_INDEX  = build_char_index(FASTA_VOCAB)

def encode_sequence(seq, index, max_len: int) -> np.ndarray:
    seq = str(seq)
    ids = [index.get(ch, 0) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)

# ------------------ Data loader ------------------
def load_csv(
    path: str,
    task_name: str,
    smiles_col: str = "SMILES",
    fasta_col: str = "FASTA",
    smiles_max_len: int = 100,
    fasta_max_len: int = 1000,
):
    """
    Load CSV and encode SMILES / FASTA.
    Label column is chosen automatically depending on task_name:
      - Kd:  prefer 'pKd', fallback to 'affinity'
      - Ki:  prefer 'pKi', fallback to 'affinity'
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Error] CSV file not found: {path}")

    df = pd.read_csv(path)

    if task_name == "Kd":
        candidate_labels = ["pKd", "affinity"]
    elif task_name == "Ki":
        candidate_labels = ["pKi", "affinity"]
    else:
        raise ValueError(f"[Error] Invalid task_name: {task_name}. Must be one of ['Kd', 'Ki'].")

    label_col = None
    for col in candidate_labels:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError(
            f"[Error] Could not find any label column among {candidate_labels}. "
            f"Available columns: {df.columns.tolist()}"
        )

    for col in [smiles_col, fasta_col]:
        if col not in df.columns:
            raise ValueError(
                f"[Error] Column '{col}' not found in CSV. "
                f"Available columns: {df.columns.tolist()}"
            )

    X_smiles = np.stack([
        encode_sequence(s, SMILES_INDEX, smiles_max_len)
        for s in df[smiles_col].values
    ])
    X_fasta = np.stack([
        encode_sequence(s, FASTA_INDEX, fasta_max_len)
        for s in df[fasta_col].values
    ])
    y = df[label_col].astype(np.float32).values

    return X_smiles, X_fasta, y

# ------------------ Model ------------------
def build_deepdta_like_model(
    smiles_len: int = 100,
    fasta_len: int = 1000,
    smiles_vocab_size: int = len(SMILES_INDEX) + 1,
    fasta_vocab_size: int = len(FASTA_INDEX) + 1,
    emb_dim: int = 128,
) -> tf.keras.Model:

    kernel_init = initializers.GlorotUniform(seed=SEED)
    bias_init = initializers.Zeros()

    inp_smi = layers.Input(shape=(smiles_len,), name="smiles")
    x = layers.Embedding(
        smiles_vocab_size,
        emb_dim,
        mask_zero=False,
        embeddings_initializer=initializers.RandomUniform(seed=SEED),
    )(inp_smi)
    x = layers.Conv1D(32, 4, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = layers.Conv1D(64, 4, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = layers.Conv1D(96, 4, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = layers.GlobalMaxPooling1D()(x)

    inp_fa = layers.Input(shape=(fasta_len,), name="fasta")
    y = layers.Embedding(
        fasta_vocab_size,
        emb_dim,
        mask_zero=False,
        embeddings_initializer=initializers.RandomUniform(seed=SEED),
    )(inp_fa)
    y = layers.Conv1D(32, 8, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(y)
    y = layers.Conv1D(64, 8, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(y)
    y = layers.Conv1D(96, 8, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(y)
    y = layers.GlobalMaxPooling1D()(y)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(1024, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(z)
    z = layers.Dropout(0.2, seed=SEED)(z)
    z = layers.Dense(1024, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(z)
    z = layers.Dropout(0.2, seed=SEED)(z)
    z = layers.Dense(512, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(z)
    out = layers.Dense(1, kernel_initializer=kernel_init, bias_initializer=bias_init)(z)

    model = models.Model(inputs=[inp_smi, inp_fa], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model

# ------------------ Plot history ------------------
def plot_history(history, out_dir: str):
    history_dict = history.history if hasattr(history, "history") else {}
    train_loss = history_dict.get("loss")
    val_loss = history_dict.get("val_loss")

    if train_loss is None:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train Loss")
    if val_loss is not None:
        plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training History")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    save_path = os.path.join(out_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ------------------ Train loop ------------------
def setup_seed_and_gpu(seed: int, gpu: str | None):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"[Info] Using GPU: {gpu}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"[Info] TF2: GPU memory growth enabled for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"[Warn] GPU memory growth setting error: {e}")
    else:
        print("[Warn] No GPU devices found. Using CPU.")

def train(args):
    setup_seed_and_gpu(args.seed, args.gpu)

    # --------------------
    # Load datasets
    # --------------------
    print(f"[Info] Loading Train data from {args.train_csv}...")
    Xs_tr, Xf_tr, y_tr = load_csv(
        args.train_csv,
        task_name=args.task_name,
        smiles_max_len=args.smiles_max_len,
        fasta_max_len=args.fasta_max_len,
    )

    print(f"[Info] Loading Validation data from {args.val_csv}...")
    Xs_va, Xf_va, y_va = load_csv(
        args.val_csv,
        task_name=args.task_name,
        smiles_max_len=args.smiles_max_len,
        fasta_max_len=args.fasta_max_len,
    )

    print(f"[Info] Loading Test data from {args.test_csv}...")
    Xs_te, Xf_te, y_te = load_csv(
        args.test_csv,
        task_name=args.task_name,
        smiles_max_len=args.smiles_max_len,
        fasta_max_len=args.fasta_max_len,
    )

    print(f"[Info] Train size: {len(y_tr)}, "
          f"Validation size: {len(y_va)}, Test size: {len(y_te)}")

    # --------------------
    # Build model
    # --------------------
    model = build_deepdta_like_model(
        smiles_len=args.smiles_max_len,
        fasta_len=args.fasta_max_len,
    )

    # --------------------
    # Callbacks
    # --------------------
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_path = os.path.join(args.out_dir, "best.h5")
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    csv_log_path = os.path.join(args.out_dir, "training_log.csv")
    csv_logger = tf.keras.callbacks.CSVLogger(csv_log_path, separator=",", append=False)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        restore_best_weights=True,
        verbose=1,
    )

    # --------------------
    # Train
    # --------------------
    print("[Info] Starting training...")
    history = model.fit(
        x={"smiles": Xs_tr, "fasta": Xf_tr}, y=y_tr,
        validation_data=({"smiles": Xs_va, "fasta": Xf_va}, y_va),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[ckpt, early_stopping, csv_logger],
        verbose=1,
    )

    # Save history + plot
    try:
        hist_df = pd.DataFrame(history.history)
        hist_csv_path = os.path.join(args.out_dir, "history.csv")
        hist_df.to_csv(hist_csv_path, index=False)
    except Exception:
        pass

    plot_history(history, args.out_dir)

    # --------------------
    # Final evaluation on test set
    # --------------------
    print("[Info] Loading best model for evaluation...")
    try:
        model.load_weights(ckpt_path)
    except FileNotFoundError:
        print("[Warn] Best model weights not found. Evaluating with current weights.")

    preds = model.predict({"smiles": Xs_te, "fasta": Xf_te}, verbose=0).reshape(-1)

    mae = mean_absolute_error(y_te, preds)
    rmse = float(np.sqrt(np.mean((y_te - preds) ** 2)))
    pcc = float(pearsonr(y_te, preds)[0]) if (len(y_te) > 1) else float("nan")

    metrics_path = os.path.join(args.out_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nPCC: {pcc:.4f}\n")

    print(f"[Info] Test MAE: {mae:.4f} | RMSE: {rmse:.4f} | PCC: {pcc:.4f}")
    print(f"[Info] Metrics saved to {metrics_path}")

# ------------------ CLI ------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, choices=["Kd", "Ki"])
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--smiles_max_len", type=int, default=100)
    parser.add_argument("--fasta_max_len", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--gpu", type=str, default=None, help="GPU device ID (e.g., '0')")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)

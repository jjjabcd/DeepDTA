import os
import sys
import warnings
import argparse
import random
import numpy as np
import pandas as pd

# --- Matplotlib (비대화형 백엔드 설정 및 임포트) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 재현성을 위한 Seed 고정 (모든 라이브러리) ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = '0'

# --- 모든 경고 억제 (TensorFlow, Python 등) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN 메시지 억제
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# TensorFlow import 전에 로깅 설정
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import initializers
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

# TensorFlow Seed 고정
tf.random.set_seed(SEED)
tf.compat.v1.set_random_seed(SEED)

# TensorFlow 로깅 억제
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

# 표준 출력/에러 리다이렉션으로 TensorFlow 메시지 억제
if hasattr(sys, '_getframe'):
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

# ------------------ Tokenizers ------------------
# (기존과 동일)
SMILES_VOCAB = list("#%)(+-.0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ@[]\\abdefghilmnoprstuy")
FASTA_VOCAB  = list("ACDEFGHIKLMNPQRSTVWY")

def build_char_index(vocab):
    return {ch: i + 1 for i, ch in enumerate(vocab)}

SMILES_INDEX = build_char_index(SMILES_VOCAB)
FASTA_INDEX  = build_char_index(FASTA_VOCAB)

def encode_sequence(seq, index, max_len):
    seq = str(seq)
    ids = [index.get(ch, 0) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)

def load_csv(path, smiles_col_candidates=("compound_iso_smiles", "SMILES", "smiles"),
             fasta_col_candidates=("target_sequence", "FASTA", "fasta", "sequence"),
             label_col="affinity",
             smiles_max_len=150,
             fasta_max_len=1000):
    # (기존과 동일)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
         raise FileNotFoundError(f"CSV file not found at: {path}")
    smiles_col = next((c for c in smiles_col_candidates if c in df.columns), None)
    fasta_col  = next((c for c in fasta_col_candidates  if c in df.columns), None)
    if smiles_col is None or fasta_col is None or label_col not in df.columns:
        raise ValueError(f"CSV must contain SMILES and FASTA and '{label_col}' columns. Got columns: {df.columns.tolist()}")
    X_smiles = np.stack([encode_sequence(s, SMILES_INDEX, smiles_max_len) for s in df[smiles_col].values])
    X_fasta  = np.stack([encode_sequence(s, FASTA_INDEX,  fasta_max_len) for s in df[fasta_col].values])
    y        = df[label_col].values.astype(np.float32)
    return X_smiles, X_fasta, y

# ------------------ Model ------------------
def build_deepdta_like_model(smiles_len=150, fasta_len=1000,
                             smiles_vocab_size=len(SMILES_INDEX) + 1,
                             fasta_vocab_size=len(FASTA_INDEX) + 1,
                             emb_dim=128):
    # Seed 고정을 위한 초기화 함수
    kernel_init = initializers.GlorotUniform(seed=SEED)
    bias_init = initializers.Zeros()
    
    inp_smi = layers.Input(shape=(smiles_len,), name="smiles")
    x = layers.Embedding(smiles_vocab_size, emb_dim, mask_zero=False, 
                        embeddings_initializer=initializers.RandomUniform(seed=SEED))(inp_smi)
    x = layers.Conv1D(32, 4, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = layers.Conv1D(64, 4, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = layers.Conv1D(96, 4, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = layers.GlobalMaxPooling1D()(x)
    inp_fa = layers.Input(shape=(fasta_len,), name="fasta")
    y = layers.Embedding(fasta_vocab_size, emb_dim, mask_zero=False,
                        embeddings_initializer=initializers.RandomUniform(seed=SEED))(inp_fa)
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

# ------------------ Utils: History Plot ------------------
def plot_history(history, out_dir):
    history_dict = history.history if hasattr(history, "history") else {}
    train_loss = history_dict.get("loss", None)
    val_loss = history_dict.get("val_loss", None)

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

# ------------------ Train ------------------
def train(args):
    # --- 재현성을 위한 추가 Seed 고정 (함수 시작 시) ---
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.compat.v1.set_random_seed(SEED)
    
    # TensorFlow deterministic 연산 활성화 (재현성 향상)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        # TF 2.8 이하 버전에서는 사용 불가
        pass
    
    # --- GPU 설정 (TF 2.x) ---
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")
    
    # TF 2.x GPU 메모리 설정
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow 2.x: GPU memory growth enabled for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"GPU memory growth setting error: {e}")
    else:
        print("Warning: No GPU devices found. Using CPU.")

    print(f"Loading Train data from {args.train_csv}...")
    Xs_train_full, Xf_train_full, y_train_full = load_csv(
        args.train_csv, label_col=args.label_col,
        smiles_max_len=args.smiles_max_len, fasta_max_len=args.fasta_max_len
    )
    print(f"Loading Test data from {args.test_csv}...")
    Xs_te, Xf_te, y_te = load_csv(
        args.test_csv, label_col=args.label_col,
        smiles_max_len=args.smiles_max_len, fasta_max_len=args.fasta_max_len
    )
    print("Splitting Train data into Train/Validation (9:1)...")
    Xs_tr, Xs_va, Xf_tr, Xf_va, y_tr, y_va = train_test_split(
        Xs_train_full, Xf_train_full, y_train_full,
        test_size=0.1, random_state=42
    )
    print(f"Train size: {len(y_tr)}, Validation size: {len(y_va)}, Test size: {len(y_te)}")

    # TF 1.15에서는 MirroredStrategy 사용이 제한적일 수 있으므로 단일 모델 빌드
    model = build_deepdta_like_model(smiles_len=args.smiles_max_len, fasta_len=args.fasta_max_len)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "best.h5")
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path, monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True, verbose=1
    )
    csv_log_path = os.path.join(args.out_dir, "training_log.csv")
    csv_logger = tf.keras.callbacks.CSVLogger(csv_log_path, separator=",", append=False)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1
    )

    print("Starting training...")
    history = model.fit(
        x={"smiles": Xs_tr, "fasta": Xf_tr}, y=y_tr,
        validation_data=({"smiles": Xs_va, "fasta": Xf_va}, y_va),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=[ckpt, early_stopping, csv_logger], verbose=1
    )

    try:
        hist_df = pd.DataFrame(history.history)
        hist_csv_path = os.path.join(args.out_dir, "history.csv")
        hist_df.to_csv(hist_csv_path, index=False)
    except Exception:
        pass
    plot_history(history, args.out_dir)

    print("Loading best model for evaluation...")
    try:
        model.load_weights(ckpt_path)
    except FileNotFoundError:
         print("Warning: Best model weights not found. Evaluating with current weights.")

    preds = model.predict({"smiles": Xs_te, "fasta": Xf_te}, verbose=0).reshape(-1)
    mae = mean_absolute_error(y_te, preds)
    rmse = float(np.sqrt(np.mean((y_te - preds) ** 2)))
    if len(y_te) > 1 and np.std(preds) > 1e-9 and np.std(y_te) > 1e-9:
        pcc = float(pearsonr(y_te, preds)[0])
    else:
        pcc = float("nan")

    metrics_path = os.path.join(args.out_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nPCC: {pcc:.4f}\n")
    print(f" Evaluation Results -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | PCC: {pcc:.4f}")
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="affinity")
    parser.add_argument("--smiles_max_len", type=int, default=150)
    parser.add_argument("--fasta_max_len", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--gpu", type=str, default=None, help="GPU device ID(s) (e.g., '0' or '1')")
    args = parser.parse_args()
    train(args)
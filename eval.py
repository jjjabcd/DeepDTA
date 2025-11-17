import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------ Tokenizers ------------------
SMILES_VOCAB = list("#%)(+-.0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ@[]\\abdefghilmnoprstuy")
FASTA_VOCAB  = list("ACDEFGHIKLMNPQRSTVWY")

def build_char_index(vocab):
    return {ch: i + 1 for i, ch in enumerate(vocab)}

SMILES_INDEX = build_char_index(SMILES_VOCAB)
FASTA_INDEX  = build_char_index(FASTA_VOCAB)

def encode_sequence(seq, index, max_len):
    seq = str(seq)
    ids = [index.get(ch, 0) for ch in seq[:max_len]]
    return np.array(ids + [0] * (max_len - len(ids)), dtype=np.int32)


# ------------------ Model ------------------
def build_deepdta_like_model(smiles_len=150, fasta_len=1000,
                             smiles_vocab_size=len(SMILES_INDEX) + 1,
                             fasta_vocab_size=len(FASTA_INDEX) + 1,
                             emb_dim=128):

    inp_smi = layers.Input(shape=(smiles_len,), name="smiles")
    x = layers.Embedding(smiles_vocab_size, emb_dim)(inp_smi)
    x = layers.Conv1D(32, 4, activation="relu")(x)
    x = layers.Conv1D(64, 4, activation="relu")(x)
    x = layers.Conv1D(96, 4, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)

    inp_fa = layers.Input(shape=(fasta_len,), name="fasta")
    y = layers.Embedding(fasta_vocab_size, emb_dim)(inp_fa)
    y = layers.Conv1D(32, 8, activation="relu")(y)
    y = layers.Conv1D(64, 8, activation="relu")(y)
    y = layers.Conv1D(96, 8, activation="relu")(y)
    y = layers.GlobalMaxPooling1D()(y)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(1024, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(1024, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(512, activation="relu")(z)
    out = layers.Dense(1)(z)

    return models.Model(inputs=[inp_smi, inp_fa], outputs=out)


# ------------------ Evaluate ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True, choices=["Kd", "Ki"])
    parser.add_argument("--smiles_max_len", type=int, default=150)
    parser.add_argument("--fasta_max_len", type=int, default=1000)
    parser.add_argument("--out_csv", type=str, default="predictions.csv")
    parser.add_argument("--out_metrics", type=str, default="metrics.txt")
    args = parser.parse_args()

    df = pd.read_csv(args.test_csv)

    # label column
    label_col = "pKd" if args.task_name == "Kd" else "pKi"

    # Encode all inputs
    Xs = np.stack([encode_sequence(s, SMILES_INDEX, args.smiles_max_len) for s in df["SMILES"]])
    Xf = np.stack([encode_sequence(s, FASTA_INDEX, args.fasta_max_len) for s in df["FASTA"]])

    # Build & load model
    model = build_deepdta_like_model(args.smiles_max_len, args.fasta_max_len)
    model.load_weights(args.weights)

    preds = model.predict({"smiles": Xs, "fasta": Xf}, verbose=0).reshape(-1)

    df["prediction_value"] = preds

    # Metrics
    y_true = df[label_col].values
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(np.mean((y_true - preds) ** 2))

    if np.std(preds) > 1e-9 and np.std(y_true) > 1e-9:
        pcc = pearsonr(y_true, preds)[0]
    else:
        pcc = float("nan")

    # Save predictions & metrics
    df.to_csv(args.out_csv, index=False)

    with open(args.out_metrics, "w") as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nPCC: {pcc:.4f}\n")

    print("=== Evaluation Done ===")
    print("Saved predictions →", args.out_csv)
    print("Saved metrics     →", args.out_metrics)
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | PCC: {pcc:.4f}")


if __name__ == "__main__":
    main()

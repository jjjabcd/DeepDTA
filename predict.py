import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd

# ------------------ Tokenizers ------------------
SMILES_VOCAB = list("#%)(+-.0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ@[]\\abdefghilmnoprstuy")
FASTA_VOCAB  = list("ACDEFGHIKLMNPQRSTVWY")

def build_char_index(vocab):
    return {ch: i + 1 for i, ch in enumerate(vocab)}   # 0 = PAD

SMILES_INDEX = build_char_index(SMILES_VOCAB)
FASTA_INDEX  = build_char_index(FASTA_VOCAB)

def encode_sequence(seq, index, max_len):
    seq = str(seq)
    ids = [index.get(ch, 0) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)

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


def predict_from_csv(test_csv, weights, out_csv, smiles_max_len=150, fasta_max_len=1000):
    df = pd.read_csv(test_csv)

    xs = np.stack([encode_sequence(s, SMILES_INDEX, smiles_max_len) for s in df["SMILES"]])
    xf = np.stack([encode_sequence(s, FASTA_INDEX, fasta_max_len) for s in df["FASTA"]])

    model = build_deepdta_like_model(smiles_max_len, fasta_max_len)
    model.load_weights(weights)

    preds = model.predict({"smiles": xs, "fasta": xf}, verbose=0).reshape(-1)

    df["predicted_value"] = preds
    df.to_csv(out_csv, index=False)

    print(f"[+] Saved predictions → {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--smiles_max_len", type=int, default=150)
    parser.add_argument("--fasta_max_len", type=int, default=1000)
    args = parser.parse_args()

    predict_from_csv(args.test_csv, args.weights, args.out_csv,
                     args.smiles_max_len, args.fasta_max_len)

if __name__ == "__main__":
    main()
import os

# Disable XLA before importing TensorFlow
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from train_from_csv import (
    encode_sequence,
    SMILES_INDEX, FASTA_INDEX,
    build_model, set_gpu
)

# Disable XLA JIT
tf.config.optimizer.set_jit(False)


def load_for_predict(csv_path, smiles_len=100, fasta_len=1000):
    """Load CSV file and encode SMILES/FASTA sequences."""
    df = pd.read_csv(csv_path)
    Xs = np.stack([encode_sequence(s, SMILES_INDEX, smiles_len) for s in df["SMILES"]])
    Xf = np.stack([encode_sequence(s, FASTA_INDEX, fasta_len) for s in df["FASTA"]])
    return df, Xs, Xf


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--smiles_max_len", type=int, default=100)
    parser.add_argument("--fasta_max_len", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2048)

    parser.add_argument("--gpu", required=True)

    args = parser.parse_args()

    # GPU setup
    set_gpu(int(args.gpu))

    print(f"[Info] Loading test data: {args.test_csv}")
    df, Xs, Xf = load_for_predict(
        args.test_csv,
        smiles_len=args.smiles_max_len,
        fasta_len=args.fasta_max_len
    )

    print("[Info] Building model...")
    model = build_model(
        smiles_len=args.smiles_max_len,
        fasta_len=args.fasta_max_len
    )

    print(f"[Info] Loading weights: {args.ckpt_path}")
    model.load_weights(args.ckpt_path)

    print("[Info] Predicting...")
    preds = model.predict(
        {"smiles": Xs, "fasta": Xf},
        batch_size=args.batch_size,
        verbose=1
    ).reshape(-1)

    df["predicted_value"] = preds
    df.to_csv(args.out_csv, index=False)

    print(f"[Info] Saved predictions → {args.out_csv}")


if __name__ == "__main__":
    main()

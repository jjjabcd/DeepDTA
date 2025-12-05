import os
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

tf.config.optimizer.set_jit(False)


def load_for_predict(csv_path, smiles_len=100, fasta_len=1000):
    df = pd.read_csv(csv_path)
    Xs = np.stack([encode_sequence(s, SMILES_INDEX, smiles_len) for s in df["SMILES"]])
    Xf = np.stack([encode_sequence(s, FASTA_INDEX, fasta_len) for s in df["FASTA"]])
    return df, Xs, Xf


def to_standard_value(p):
    """Convert pKi/pKd to standard_value using x = 10^(9 - p)."""
    if pd.isna(p):
        return None
    return 10 ** (9 - p)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--ckpt_root", required=True)
    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--smiles_max_len", type=int, default=100)
    parser.add_argument("--fasta_max_len", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--gpu", required=True)

    args = parser.parse_args()
    set_gpu(int(args.gpu))

    print(f"[Info] Loading test CSV: {args.test_csv}")
    df, Xs, Xf = load_for_predict(
        args.test_csv,
        smiles_len=args.smiles_max_len,
        fasta_len=args.fasta_max_len
    )

    target_col = "pKi" if args.task_name == "Ki" else "pKd"

    fold_preds = []

    for fold in [1, 2, 3]:
        ckpt_path = f"{args.ckpt_root}/fold_{fold}/best.h5"
        fold_col = f"fold{fold}_pred"

        print(f"\n[Info] Loading fold {fold} checkpoint: {ckpt_path}")
        model = build_model(
            smiles_len=args.smiles_max_len,
            fasta_len=args.fasta_max_len
        )
        model.load_weights(ckpt_path)

        print(f"[Info] Predicting fold {fold}...")
        preds = model.predict(
            {"smiles": Xs, "fasta": Xf},
            batch_size=args.batch_size,
            verbose=1
        ).reshape(-1)

        df[fold_col] = preds
        fold_preds.append(fold_col)

    df[target_col] = df[target_col].fillna(
        df[fold_preds].mean(axis=1)
    )

    df["standard_value"] = df[target_col].apply(to_standard_value)

    df.to_csv(args.out_csv, index=False)
    print(f"\n[Info] Saved filled predictions → {args.out_csv}")


if __name__ == "__main__":
    main()

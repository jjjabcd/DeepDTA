import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from train_from_csv import (
    load_csv,
    build_deepdta_like_model,
)

def setup_gpu(gpu: str | None):
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

def predict(args):
    setup_gpu(args.gpu)

    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"[Error] test_csv not found: {args.test_csv}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"[Error] weights file not found: {args.weights}")

    print(f"[Info] Loading Test data from {args.test_csv}...")
    Xs_te, Xf_te, y_te = load_csv(
        args.test_csv,
        task_name=args.task_name,
        smiles_max_len=args.smiles_max_len,
        fasta_max_len=args.fasta_max_len,
    )

    df_test = pd.read_csv(args.test_csv)

    print("[Info] Building model and loading weights...")
    model = build_deepdta_like_model(
        smiles_len=args.smiles_max_len,
        fasta_len=args.fasta_max_len,
    )
    model.load_weights(args.weights)

    print("[Info] Running prediction on test set...")
    preds = model.predict(
        {"smiles": Xs_te, "fasta": Xf_te},
        batch_size=args.batch_size,
        verbose=0,
    ).reshape(-1)

    df_out = df_test.copy()
    df_out["prediction_value"] = preds

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"[Info] Saved predictions → {args.out_csv}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, choices=["Kd", "Ki"])
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--smiles_max_len", type=int, default=100)
    parser.add_argument("--fasta_max_len", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--gpu", type=str, default=None, help="GPU device ID (e.g., '0')")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    predict(args)

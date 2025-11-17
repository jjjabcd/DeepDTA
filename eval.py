import os
import argparse
import pandas as pd

from src.metric import get_rmse, get_pcc, get_cindex, get_rm2

def evaluate(args):
    if not os.path.exists(args.pred_csv):
        raise FileNotFoundError(f"[Error] pred_csv not found: {args.pred_csv}")

    df = pd.read_csv(args.pred_csv)

    # Determine label column
    if args.task_name == "Kd":
        candidates = ["pKd", "affinity"]
    elif args.task_name == "Ki":
        candidates = ["pKi", "affinity"]
    else:
        raise ValueError(f"[Error] Invalid task_name: {args.task_name}")

    label_col = None
    for c in candidates:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        raise ValueError(
            f"[Error] Could not find label column among {candidates}. "
            f"Available columns: {df.columns.tolist()}"
        )

    if "prediction_value" not in df.columns:
        raise ValueError("[Error] Column 'prediction_value' not found in pred_csv.")

    y_true = df[label_col].values
    y_pred = df["prediction_value"].values

    rmse = float(get_rmse(y_true, y_pred))
    pcc  = float(get_pcc(y_true, y_pred))
    cidx = float(get_cindex(y_true, y_pred))
    rm2  = float(get_rm2(y_true, y_pred))

    metrics_df = pd.DataFrame(
        [{
            "task_name": args.task_name,
            "n_samples": len(y_true),
            "rmse": rmse,
            "pcc": pcc,
            "cindex": cidx,
            "rm2": rm2,
        }]
    )

    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    metrics_df.to_csv(args.out_metrics, index=False)

    print(f"[Info] RMSE   : {rmse:.4f}")
    print(f"[Info] PCC    : {pcc:.4f}")
    print(f"[Info] C-index: {cidx:.4f}")
    print(f"[Info] RM2    : {rm2:.4f}")
    print(f"[Info] Metrics saved → {args.out_metrics}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, choices=["Kd", "Ki"])
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--out_metrics", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    evaluate(args)

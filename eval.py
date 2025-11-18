import argparse
import pandas as pd
from src.metric import (
    get_mse, get_rmse, get_pcc, get_rm2, get_cindex
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--task_name", required=True)   # Kd or Ki
    parser.add_argument("--out_metrics", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)

    label_col = "pKd" if args.task_name == "Kd" else "pKi"

    y = df[label_col].values
    p = df["predicted_value"].values

    mse  = get_mse(y, p)
    rmse = get_rmse(y, p)
    pcc  = get_pcc(y, p)
    ci   = get_cindex(y, p)
    rm2  = get_rm2(y, p)

    out = pd.DataFrame([{
        "MSE": mse,
        "RMSE": rmse,
        "PCC": pcc,
        "CI": ci,
        "RM2": rm2
    }])

    out.to_csv(args.out_metrics, index=False)
    print(f"Saved evaluation → {args.out_metrics}")


if __name__ == "__main__":
    main()

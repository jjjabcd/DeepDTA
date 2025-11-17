#!/bin/bash

FOLD=$1

if [ -z "$FOLD" ]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 [FOLD_NUMBER]"
  exit 1
fi

TASK_NAME="Kd"
OUT_DIR="./results/${TASK_NAME}/fold_${FOLD}"
PRED_CSV="${OUT_DIR}/predictions.csv"
OUT_METRICS="${OUT_DIR}/metrics.csv"

echo "===== Evaluating Fold ${FOLD} ====="
echo " Pred CSV    : ${PRED_CSV}"
echo " Metrics CSV : ${OUT_METRICS}"

python eval.py \
    --task_name "${TASK_NAME}" \
    --pred_csv "${PRED_CSV}" \
    --out_metrics "${OUT_METRICS}"

echo "===== Fold ${FOLD} Evaluation Finished ====="

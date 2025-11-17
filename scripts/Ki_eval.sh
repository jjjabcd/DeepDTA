#!/bin/bash

FOLD=$1

if [ -z "$FOLD" ]; then
  echo "Usage: $0 [FOLD_NUMBER]"
  exit 1
fi

TASK_NAME="Ki"
ROOT="/home/rlawlsgurjh/hdd/work/ChEMBL/data/processed/${TASK_NAME}/fold${FOLD}"
TEST_CSV="${ROOT}/test.csv"

OUT_DIR="./results/${TASK_NAME}/fold_${FOLD}"
PRED_CSV="${OUT_DIR}/predictions.csv"
OUT_METRIC="${OUT_DIR}/eval_metrics.csv"

echo "===== Evaluating Fold ${FOLD} ====="

python eval.py \
    --task_name "${TASK_NAME}" \
    --test_csv "${TEST_CSV}" \
    --pred_csv "${PRED_CSV}" \
    --out_csv "${OUT_METRIC}"

echo "Saved evaluation → ${OUT_METRIC}"
echo "===== Fold ${FOLD} Evaluation Finished ====="

#!/bin/bash

FOLD=$1

TASK_NAME="Kd"

DATA_ROOT="/home/rlawlsgurjh/hdd/work/ChEMBL/data/processed/${TASK_NAME}/fold${FOLD}"
TEST_CSV="${DATA_ROOT}/test.csv"

OUT_DIR="./results/${TASK_NAME}/fold_${FOLD}"
PRED_CSV="${OUT_DIR}/predictions.csv"
OUT_METRICS="${OUT_DIR}/eval_metrics.csv"

echo "===== Evaluating Fold ${FOLD} ====="

python eval.py \
    --test_csv "${TEST_CSV}" \
    --pred_csv "${PRED_CSV}" \
    --task_name "${TASK_NAME}" \
    --out_metrics "${OUT_METRICS}"

echo "Saved evaluation → ${OUT_METRICS}"

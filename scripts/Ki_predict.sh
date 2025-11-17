#!/bin/bash

FOLD=$1
GPU_ID=$2

if [ -z "$FOLD" ] || [ -z "$GPU_ID" ]; then
  echo "Usage: $0 [FOLD_NUMBER] [GPU_ID]"
  exit 1
fi

TASK_NAME="Ki"
ROOT="/home/rlawlsgurjh/hdd/work/ChEMBL/data/processed/${TASK_NAME}/fold${FOLD}"
TEST_CSV="${ROOT}/test.csv"

OUT_DIR="./results/${TASK_NAME}/fold_${FOLD}"
WEIGHTS="${OUT_DIR}/best.h5"
OUT_PRED="${OUT_DIR}/predictions.csv"

echo "===== Predicting Fold ${FOLD} ====="
echo "Weight: ${WEIGHTS}"
echo "Test  : ${TEST_CSV}"

python predict.py \
    --weights "${WEIGHTS}" \
    --test_csv "${TEST_CSV}" \
    --task_name "${TASK_NAME}" \
    --out_csv "${OUT_PRED}"

echo "Saved predictions → ${OUT_PRED}"
echo "===== Fold ${FOLD} Prediction Finished ====="

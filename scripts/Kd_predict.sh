#!/bin/bash

FOLD=$1
GPU_ID=$2

TASK_NAME="Kd"

DATA_ROOT="/home/rlawlsgurjh/hdd/work/ChEMBL/data/processed/${TASK_NAME}/fold${FOLD}"
TEST_CSV="${DATA_ROOT}/test.csv"

OUT_DIR="./results/${TASK_NAME}/fold_${FOLD}"
WEIGHTS="${OUT_DIR}/best.h5"
OUT_PRED="${OUT_DIR}/predictions.csv"

echo "===== Predicting Fold ${FOLD} ====="
echo "Weights: ${WEIGHTS}"
echo "Test   : ${TEST_CSV}"

python predict.py \
    --test_csv "${TEST_CSV}" \
    --weights "${WEIGHTS}" \
    --task_name "${TASK_NAME}" \
    --out_csv "${OUT_PRED}"

echo "Saved predictions → ${OUT_PRED}"

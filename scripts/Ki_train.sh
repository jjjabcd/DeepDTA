#!/bin/bash

FOLD=$1
GPU_ID=$2

if [ -z "$FOLD" ] || [ -z "$GPU_ID" ]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 [FOLD_NUMBER] [GPU_ID]"
  exit 1
fi

TASK_NAME="Ki"
ROOT="/home/rlawlsgurjh/hdd/work/ChEMBL/data/processed/${TASK_NAME}/fold${FOLD_NUM}"
TRAIN_CSV="${ROOT}/train.csv"
VAL_CSV="${ROOT}/val.csv"
TEST_CSV="${ROOT}/test.csv"

OUT_DIR="./results/${TASK_NAME}/fold_${FOLD}"

echo "--- Starting Fold ${FOLD} on GPU ${GPU_ID} ---"
echo "Train file: ${TRAIN_CSV}"
echo "Test file: ${TEST_CSV}"
echo "Output dir: ${OUT_DIR}"

TF_CPP_MIN_LOG_LEVEL=3 \
TF_ENABLE_ONEDNN_OPTS=0 \
PYTHONWARNINGS=ignore \
python train_from_csv.py \
    --task_name "${TASK_NAME}" \
    --train_csv "${TRAIN_CSV}" \
    --val_csv "${VAL_CSV}" \
    --test_csv "${TEST_CSV}" \
    --label_col affinity \
    --out_dir "${OUT_DIR}" \
    --epochs 1000 \
    --batch_size 256 \
    --seed 42 \
    --patience 10 \
    --gpu "${GPU_ID}" 2>/dev/null

echo "Fold ${FOLD} finished on GPU ${GPU_ID}."
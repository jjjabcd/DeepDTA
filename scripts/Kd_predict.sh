#!/bin/bash

FOLD=$1
GPU_ID=$2

if [ -z "$FOLD" ] || [ -z "$GPU_ID" ]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 [FOLD_NUMBER] [GPU_ID]"
  exit 1
fi

TASK_NAME="Kd"
ROOT="/home/rlawlsgurjh/hdd/work/ChEMBL/data/processed/${TASK_NAME}/fold${FOLD}"
TEST_CSV="${ROOT}/test.csv"

OUT_DIR="./results/${TASK_NAME}/fold_${FOLD}"
WEIGHTS="${OUT_DIR}/best.h5"
OUT_CSV="${OUT_DIR}/predictions.csv"

echo "===== Predicting Fold ${FOLD} ====="
echo " Weight: ${WEIGHTS}"
echo " Test  : ${TEST_CSV}"
echo " Out   : ${OUT_CSV}"

TF_CPP_MIN_LOG_LEVEL=3 \
TF_ENABLE_ONEDNN_OPTS=0 \
PYTHONWARNINGS=ignore \
python predict.py \
    --task_name "${TASK_NAME}" \
    --test_csv "${TEST_CSV}" \
    --weights "${WEIGHTS}" \
    --smiles_max_len 100 \
    --fasta_max_len 1000 \
    --batch_size 256 \
    --out_csv "${OUT_CSV}" \
    --gpu "${GPU_ID}"

echo "===== Fold ${FOLD} Prediction Finished ====="

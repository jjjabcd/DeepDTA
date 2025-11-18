#!/bin/bash

# Disable unnecessary TF logs and XLA
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export TF_XLA_FLAGS=--tf_xla_auto_jit=0
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

FOLD=$1
GPU_ID=$2

if [ -z "$FOLD" ] || [ -z "$GPU_ID" ]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 [FOLD_NUMBER] [GPU_ID]"
  exit 1
fi

TASK_NAME="Ki"
ROOT="/home/rlawlsgurjh/hdd/work/ChEMBL/data/processed/${TASK_NAME}/fold${FOLD}"

TEST_CSV="${ROOT}/test.csv"

OUT_DIR="./results/${TASK_NAME}/fold_${FOLD}"
CKPT_PATH="${OUT_DIR}/best.h5"
OUT_CSV="${OUT_DIR}/predictions.csv"

echo "===== Predicting Fold ${FOLD} ====="
echo " Checkpoint: ${CKPT_PATH}"
echo " Test CSV : ${TEST_CSV}"
echo " Output CSV: ${OUT_CSV}"

python predict.py \
    --task_name "${TASK_NAME}" \
    --test_csv "${TEST_CSV}" \
    --ckpt_path "${CKPT_PATH}" \
    --smiles_max_len 100 \
    --fasta_max_len 1000 \
    --batch_size 2048 \
    --out_csv "${OUT_CSV}" \
    --gpu "${GPU_ID}"

echo "===== Fold ${FOLD} Prediction Finished ====="

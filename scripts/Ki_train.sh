#!/bin/bash

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
TRAIN_CSV="${ROOT}/train.csv"
VAL_CSV="${ROOT}/val.csv"
TEST_CSV="${ROOT}/test.csv"

OUT_DIR="./results/${TASK_NAME}/fold_${FOLD}"

echo "--- Starting Fold ${FOLD} on GPU ${GPU_ID} ---"
echo " Train file: ${TRAIN_CSV}"
echo " Val file  : ${VAL_CSV}"
echo " Test file : ${TEST_CSV}"
echo " Output dir: ${OUT_DIR}"

python train_from_csv.py \
    --task_name "${TASK_NAME}" \
    --train_csv "${TRAIN_CSV}" \
    --val_csv "${VAL_CSV}" \
    --test_csv "${TEST_CSV}" \
    --out_dir "${OUT_DIR}" \
    --epochs 1000 \
    --batch_size 1024 \
    --seed 42 \
    --patience 10 \
    --smiles_max_len 100 \
    --fasta_max_len 1000 \
    --gpu "${GPU_ID}"

echo "--- Fold ${FOLD} training finished on GPU ${GPU_ID} ---"

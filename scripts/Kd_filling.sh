#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export TF_XLA_FLAGS=--tf_xla_auto_jit=0

GPU_ID=$1

if [ -z "$GPU_ID" ]; then
  echo "Usage: $0 [GPU_ID]"
  exit 1
fi

TASK_NAME="Kd"
ROOT="/home/rlawlsgurjh/hdd/work/ChEMBL/data/processed/${TASK_NAME}"
TEST_CSV="${ROOT}/${TASK_NAME}_none_only_pairs.csv"

CKPT_ROOT="./results/${TASK_NAME}"
OUT_CSV="${ROOT}/${TASK_NAME}_filled_predictions.csv"

python filling.py \
    --task_name "${TASK_NAME}" \
    --test_csv "${TEST_CSV}" \
    --ckpt_root "${CKPT_ROOT}" \
    --out_csv "${OUT_CSV}" \
    --gpu "${GPU_ID}"

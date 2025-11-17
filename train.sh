#!/bin/bash

FOLD=$1
GPU_ID=$2  # [수정] $1이 아니라 $2여야 합니다.

# 인자 확인
if [ -z "$FOLD" ] || [ -z "$GPU_ID" ]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 [FOLD_NUMBER] [GPU_ID]"
  exit 1
fi

# 기본 디렉토리 (실제 경로 확인 필요!)
TASK_NAME="Kd"
DATA_DIR="../data/processed/${TASK_NAME}/folds"

# 입력 파일 경로
TRAIN_CSV="${DATA_DIR}/DTA_train_known_fold_${FOLD}.csv"
TEST_CSV="${DATA_DIR}/DTA_test_known_fold_${FOLD}.csv"

# 출력 디렉토리
OUT_DIR="./results/${TASK_NAME}/fold_${FOLD}"

echo "--- Starting Fold ${FOLD} on GPU ${GPU_ID} ---"
echo "Train file: ${TRAIN_CSV}"
echo "Test file: ${TEST_CSV}"
echo "Output dir: ${OUT_DIR}"

# Python 스크립트 실행 (경고 메시지 억제)
TF_CPP_MIN_LOG_LEVEL=3 \
TF_ENABLE_ONEDNN_OPTS=0 \
PYTHONWARNINGS=ignore \
python train_from_csv.py \
    --train_csv "${TRAIN_CSV}" \
    --test_csv "${TEST_CSV}" \
    --label_col affinity \
    --out_dir "${OUT_DIR}" \
    --epochs 1000 \
    --batch_size 256 \
    --seed 42 \
    --patience 15 \
    --gpu "${GPU_ID}" 2>/dev/null

echo "Fold ${FOLD} finished on GPU ${GPU_ID}."
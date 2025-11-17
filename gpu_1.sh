#!/bin/bash

LOG_FILE="gpu_1.log"

nohup bash -c '
  echo "===== GPU_1 Training Sequence Started at $(date) ====="
  bash train.sh 1 0
  bash train.sh 2 0
  bash train.sh 3 0
  bash Ki_train.sh 1 0
  bash Ki_train.sh 2 0
  echo "===== GPU_1 Training Sequence Finished at $(date) ====="
' > "$LOG_FILE" 2>&1 &

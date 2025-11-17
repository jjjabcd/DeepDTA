#!/bin/bash

LOG_FILE="gpu_2.log"

nohup bash -c '
  echo "===== GPU_2 Training Sequence Started at $(date) ====="
  bash train.sh 4 1
  bash train.sh 5 1
  bash Ki_train.sh 3 1
  bash Ki_train.sh 4 1
  bash Ki_train.sh 5 1
  echo "===== GPU_2 Training Sequence Finished at $(date) ====="
' > "$LOG_FILE" 2>&1 &

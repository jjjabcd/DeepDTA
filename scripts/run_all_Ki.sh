#!/bin/bash

LOG_FILE="run_all_Ki.log"

nohup bash -c '

echo "===== Kd All-Fold Pipeline Started at $(date) ====="

for FOLD in 1 2 3
do
  echo "--- FOLD ${FOLD} START ---"

  bash Ki_train.sh  ${FOLD} 1
  bash Ki_predict.sh ${FOLD} 1
  bash Ki_eval.sh    ${FOLD}

  echo "--- FOLD ${FOLD} DONE ---"
done

echo "===== Kd All-Fold Pipeline Finished at $(date) ====="

' > "$LOG_FILE" 2>&1 &

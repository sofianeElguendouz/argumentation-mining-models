#!/usr/bin/env bash

INPUT_DIR=./data/neoplasm/
OUTPUT_DIR=./output
TASK_TYPE=rel-class
MODEL=bert
CACHE_DIR=./cache
EVALUATION_SPLIT=test
EPOCHS=3
LOG_STEPS=50
SAVE_STEPS=100
RANDOM_SEED=42

python ./run_acta.py \
  --input-dir $INPUT_DIR \
  --output-dir $OUTPUT_DIR \
  --task-type $TASK_TYPE \
  --model $MODEL \
  --cache-dir $CACHE_DIR \
  --evaluation-split $EVALUATION_SPLIT \
  --train \
  --eval-all-checkpoints \
  --epochs $EPOCHS \
  --log-every-n-steps $LOG_STEPS \
  --save-every-n-steps $SAVE_STEPS \
  --lower-case \
  --overwrite-output \
  --validation \
  --random-seed $RANDOM_SEED

#!/usr/bin/env bash

INPUT_DIR=./data/neoplasm/
OUTPUT_DIR=./output
TASK_TYPE=rel-class
MODEL=bert
CACHE_DIR=./cache
EVALUATION_SPLIT=test
EPOCHS=3
BATCH_SIZE=8
MAX_SEQ_LENGTH=64
NUM_DEVICES=1  # Since this scripts trains and evaluate, the number of devices should be 1 when non CPU
NUM_WORKERS=-1
LOG_STEPS=50
SAVE_STEPS=100
RANDOM_SEED=42

python ./run_acta.py \
  --input-dir $INPUT_DIR \
  --output-dir $OUTPUT_DIR \
  --task-type $TASK_TYPE \
  --model $MODEL \
  --cache-dir $CACHE_DIR \
  --validation \
  --train \
  --evaluation-split $EVALUATION_SPLIT \
  --num-devices $NUM_DEVICES \
  --num-workers $NUM_WORKERS \
  --epochs $EPOCHS \
  --train-batch-size $BATCH_SIZE \
  --eval-batch-size $BATCH_SIZE \
  --max-seq-length $MAX_SEQ_LENGTH \
  --lower-case \
  --log-every-n-steps $LOG_STEPS \
  --save-every-n-steps $SAVE_STEPS \
  --eval-all-checkpoints \
  --overwrite-output \
  --random-seed $RANDOM_SEED

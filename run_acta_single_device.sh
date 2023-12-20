#!/usr/bin/env bash

set -ex

# This scripts runs ACTA train and evaluation in the same process, but only uses
# a single device to avoid inconsistencies when evaluating

INPUT_DIR=./data/neoplasm/
OUTPUT_DIR=./output
CHECKPOINT_PATH=checkpoints
TASK_TYPE=seq-tag
MODEL=bert
CACHE_DIR=./cache
EVALUATION_SPLIT=test
EPOCHS=3
BATCH_SIZE=8
MAX_SEQ_LENGTH=256
LEARNING_RATE=2e-5
LABELS="O B-Claim I-Claim B-Premise I-Premise"
NUM_DEVICES=1
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
  --checkpoint-path $CHECKPOINT_PATH \
  --train \
  --evaluation-split $EVALUATION_SPLIT \
  --validation \
  --num-devices $NUM_DEVICES \
  --num-workers $NUM_WORKERS \
  --epochs $EPOCHS \
  --train-batch-size $BATCH_SIZE \
  --eval-batch-size $BATCH_SIZE \
  --max-seq-length $MAX_SEQ_LENGTH \
  --learning-rate $LEARNING_RATE \
  --labels $LABELS \
  --lower-case \
  --log-every-n-steps $LOG_STEPS \
  --save-every-n-steps $SAVE_STEPS \
  --eval-all-checkpoints \
  --overwrite-output \
  --random-seed $RANDOM_SEED

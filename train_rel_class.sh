#!/usr/bin/env bash

set -ex

TRAIN_FILE=./data/neoplasm/train_relations.tsv
VALIDATION_FILE=./data/neoplasm/dev_relations.tsv
OUTPUT_DIR=./output
TASK_TYPE=rel-class
MODEL=bert
EXPERIMENT_NAME="neoplasm"
LABELS="noRel Attack Support"

EPOCHS=5
EARLY_STOP=2
BATCH_SIZE=8
GRADIENT_ACCUMULATION=1
MAX_GRAD=1
MAX_SEQ_LENGTH=128
LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-6
WARMUP_STEPS=0
LOG_STEPS=100
SAVE_STEPS=250
RANDOM_SEED=42

python ./scripts/train.py \
  --train-data $TRAIN_FILE \
  --validation-data $VALIDATION_FILE \
  --output-dir $OUTPUT_DIR \
  --task-type $TASK_TYPE \
  --model $MODEL \
  --experiment-name "$EXPERIMENT_NAME" \
  --validation \
  --labels $LABELS \
  --num-devices -1 \
  --num-workers -1 \
  --epochs $EPOCHS \
  --early-stopping $EARLY_STOP \
  --batch-size $BATCH_SIZE \
  --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
  --max-grad-norm $MAX_GRAD \
  --max-seq-length $MAX_SEQ_LENGTH \
  --lower-case \
  --learning-rate $LEARNING_RATE \
  --weight-decay $WEIGHT_DECAY \
  --warmup-steps $WARMUP_STEPS \
  --weighted-loss \
  --log-every-n-steps $LOG_STEPS \
  --save-every-n-steps $SAVE_STEPS \
  --random-seed $RANDOM_SEED

#!/usr/bin/env bash

set -ex

TRAIN_FILE=./data/classification/touche-train.tsv
TEST_FILE=./data/classification/touche-test.tsv
VALIDATION_FILE=./data/classification/touche-validation.tsv
OUTPUT_DIR=./output
TASK_TYPE=sta-class
MODEL=deberta-v3
EXPERIMENT_NAME=touche23-valueeval
RUN_NAME=deberta-v3-model
LABELS="Position Attack Support"
RELEVANT_LABELS="Attack Support"

EPOCHS=5
EARLY_STOP=2
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
GRADIENT_ACCUMULATION=1
MAX_GRAD=1
MAX_SEQ_LENGTH=128
LEARNING_RATE=2e-5
WEIGHT_DECAY=0
WARMUP_STEPS=0
LOG_STEPS=90
SAVE_STEPS=180
RANDOM_SEED=42

python ./scripts/train.py \
  --train-data $TRAIN_FILE \
  --validation-data $VALIDATION_FILE \
  --output-dir $OUTPUT_DIR \
  --task-type $TASK_TYPE \
  --model $MODEL \
  --experiment-name $EXPERIMENT_NAME \
  --run-name $RUN_NAME \
  --labels $LABELS \
  --num-devices -1 \
  --num-workers -1 \
  --epochs $EPOCHS \
  --early-stopping $EARLY_STOP \
  --batch-size $TRAIN_BATCH_SIZE \
  --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
  --max-grad-norm $MAX_GRAD \
  --max-seq-length $MAX_SEQ_LENGTH \
  --learning-rate $LEARNING_RATE \
  --weight-decay $WEIGHT_DECAY \
  --warmup-steps $WARMUP_STEPS \
  --weighted-loss \
  --log-every-n-steps $LOG_STEPS \
  --save-every-n-steps $SAVE_STEPS \
  --random-seed $RANDOM_SEED

python ./scripts/eval.py \
  --test-data $TEST_FILE \
  --output-dir $OUTPUT_DIR \
  --task-type $TASK_TYPE \
  --model $MODEL \
  --experiment-name $EXPERIMENT_NAME \
  --run-name $RUN_NAME \
  --eval-all-checkpoints \
  --labels $LABELS \
  --relevant-labels $RELEVANT_LABELS \
  --num-workers -1 \
  --batch-size $EVAL_BATCH_SIZE \
  --max-seq-length $MAX_SEQ_LENGTH \
  --weighted-loss \
  --random-seed $RANDOM_SEED

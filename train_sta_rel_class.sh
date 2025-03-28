#!/usr/bin/env bash

set -ex

source .venv/bin/activate

TRAIN_FILE=../data/statements/relation/all-train.tsv
TEST_FILE=../data/statements/relation/touche-test.tsv
VALIDATION_FILE=../data/statements/relation/all-validation.tsv
OUTPUT_DIR=./results
TASK_TYPE=rel-class
EXPERIMENT_NAME="statements"
MODEL="deberta-v3"
RUN_NAME=deberta-v3
LABELS="noRel Attack Support"
RELEVANT_LABELS="Attack Support"

EPOCHS=3
EARLY_STOP=$EPOCHS
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=32
MAX_SEQ_LENGTH=256  # Required for better reach
GRADIENT_ACCUMULATION=1
MAX_GRAD=1
LEARNING_RATE=1e-4
WEIGHT_DECAY=0
WARMUP_STEPS=0
LOG_STEPS=2502
SAVE_STEPS=7506
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
  --random-seed $RANDOM_SEED

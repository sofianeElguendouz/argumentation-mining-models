#!/usr/bin/env bash

set -ex

TRAIN_FILE=./data/neoplasm/train_relations.tsv
TEST_FILE=./data/neoplasm/test_relations.tsv
VALIDATION_FILE=./data/neoplasm/dev_relations.tsv
OUTPUT_DIR=./results
TASK_TYPE=rel-class
MODEL=deberta-v3
EXPERIMENT_NAME="neoplasm"
RUN_NAME="deberta-v3-model"
LABELS="noRel Attack Support"
RELEVANT_LABELS="Attack Support"

EPOCHS=5
EARLY_STOP=2
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=32
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
  --run-name "$RUN_NAME" \
  --labels $LABELS \
  --num-devices -1 \
  --num-workers -1 \
  --epochs $EPOCHS \
  --early-stopping $EARLY_STOP \
  --batch-size $TRAIN_BATCH_SIZE \
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

python ./scripts/eval.py \
  --test-data $TEST_FILE \
  --output-dir $OUTPUT_DIR \
  --task-type $TASK_TYPE \
  --model $MODEL \
  --experiment-name "$EXPERIMENT_NAME" \
  --run-name "$RUN_NAME" \
  --eval-all-checkpoints \
  --labels $LABELS \
  --relevant-labels $RELEVANT_LABELS \
  --num-workers -1 \
  --batch-size $EVAL_BATCH_SIZE \
  --max-seq-length $MAX_SEQ_LENGTH \
  --lower-case \
  --weighted-loss \
  --random-seed $RANDOM_SEED

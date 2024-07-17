#!/usr/bin/env bash

set -ex

TRAIN_FILE=./data/sequence/disputool-train.conll
TEST_FILE=./data/sequence/disputool-test.conll
VALIDATION_FILE=./data/sequence/disputool-validation.conll
OUTPUT_DIR=./output
TASK_TYPE=seq-tag
MODEL=deberta-v3
EXPERIMENT_NAME=disputool
RUN_NAME=deberta-v3-model
LABELS="PAD O B-Claim I-Claim B-Premise I-Premise"
RELEVANT_LABELS="O B-Claim I-Claim B-Premise I-Premise"

EPOCHS=5
EARLY_STOP=2
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=64
GRADIENT_ACCUMULATION=1
MAX_GRAD=1
MAX_SEQ_LENGTH=10
LEARNING_RATE=1e-4
WEIGHT_DECAY=0
WARMUP_STEPS=0
LOG_STEPS=415
SAVE_STEPS=930
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

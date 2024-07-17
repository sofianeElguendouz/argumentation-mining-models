#!/usr/bin/env bash

set -ex

TEST_FILE=./data/sequence/disputool-test.conll
OUTPUT_DIR=./output
TASK_TYPE=seq-tag
MODEL=deberta-v3
EXPERIMENT_NAME=disputool
RUN_NAME=deberta-v3-model
LABELS="PAD O B-Claim I-Claim B-Premise I-Premise"
RELEVANT_LABELS="O B-Claim I-Claim B-Premise I-Premise"

BATCH_SIZE=64
MAX_SEQ_LENGTH=10
RANDOM_SEED=42

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
  --batch-size $BATCH_SIZE \
  --max-seq-length $MAX_SEQ_LENGTH \
  --random-seed $RANDOM_SEED

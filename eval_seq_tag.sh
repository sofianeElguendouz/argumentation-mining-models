#!/usr/bin/env bash

set -ex

TEST_FILE=./data/neoplasm/test.conll
OUTPUT_DIR=./results
TASK_TYPE=seq-tag
MODEL=deberta-v3
EXPERIMENT_NAME=neoplasm
RUN_NAME=deberta-v3-model
LABELS="PAD O B-Claim I-Claim B-Premise I-Premise"
RELEVANT_LABELS="O B-Claim I-Claim B-Premise I-Premise"

BATCH_SIZE=32
MAX_SEQ_LENGTH=128
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
  --lower-case \
  --random-seed $RANDOM_SEED

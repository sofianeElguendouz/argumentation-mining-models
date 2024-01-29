#!/usr/bin/env bash

set -ex

TEST_FILE=./data/neoplasm/test_relations.tsv
OUTPUT_DIR=./output
TASK_TYPE=rel-class
MODEL=bert
EXPERIMENT_NAME="neoplasm"
LABELS="noRel Attack Support"
RELEVANT_LABELS="Attack Support"

BATCH_SIZE=16
MAX_SEQ_LENGTH=128
RANDOM_SEED=42

python ./scripts/eval.py \
  --test-data $TEST_FILE \
  --output-dir $OUTPUT_DIR \
  --task-type $TASK_TYPE \
  --model $MODEL \
  --experiment-name "$EXPERIMENT_NAME" \
  --eval-all-checkpoints \
  --labels $LABELS \
  --relevant-labels $RELEVANT_LABELS \
  --num-workers -1 \
  --batch-size $BATCH_SIZE \
  --max-seq-length $MAX_SEQ_LENGTH \
  --lower-case \
  --random-seed $RANDOM_SEED

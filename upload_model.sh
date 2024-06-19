#!/usr/bin/env bash

# Example script on how to upload a trained model to Hugging Face. In this case
# it would be the model trained with the example `train_seq_tag.sh` file. See
# that the values for `--model`, `--experiment-name` and `--run-name` are the
# same ones as the ones in `train_seq_tag.sh`.

# You are required to either pass the Hugging Face token via the `--hf-token`
# option or by the `HF_TOKEN` environment variable.

set -ex

MLFLOW_DIR=./results
TASK_TYPE=seq-tag
MODEL=deberta-v3
EXPERIMENT_NAME=neoplasm
RUN_NAME=deberta-v3-model

HF_REPOSITORY="${MODEL}-${TASK_TYPE}-neoplasm"

python ./scripts/upload_model.py \
  --hf-repository $HF_REPOSITORY \
  --mlflow-dir $MLFLOW_DIR \
  --task-type $TASK_TYPE \
  --model $MODEL \
  --experiment-name $EXPERIMENT_NAME \
  --run-name $RUN_NAME \
  --hf-private-repository \
  --hf-commit-message "Uploaded ${MODEL}/${TASK_TYPE} model trained on neoplasm"

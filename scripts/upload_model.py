#!/usr/bin/env python
"""
Script to upload a model trained with AMTM (either for Relation Classification
or for Sequence Tagging) to Hugging Face to make it available via the
transformers "Pipeline" module.

    Argumentation Mining Transformers Module Training Script
    Copyright (C) 2024 Cristian Cardellino

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import logging
import mlflow
import os
import sys

from huggingface_hub import list_models
from pathlib import Path
from transformers import AutoTokenizer

from amtm.models import RelationClassificationTransformerModule, SequenceTaggingTransformerModule


logger = logging.getLogger(__name__)

MODELS = {
    "bert": "bert-base-uncased",
    "deberta-v3": "microsoft/deberta-v3-base",
    "roberta": "roberta-base",
    "tiny-bert": "prajjwal1/bert-tiny",  # Useful for debug purposes
}

# Available tasks to work with
TASKS = {
    "rel-class": RelationClassificationTransformerModule,
    "seq-tag": SequenceTaggingTransformerModule,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf-repository",
        required=True,
        help="The Hugging Face repository to upload the model. You must have write access to it.",
    )
    parser.add_argument(
        "--mlflow-dir",
        required=True,
        type=Path,
        help=(
            "The directory where the MLFlow artifacts where saved to retrieve the checkpoint "
            "file that will be uploaded to Hugging Face"
        ),
    )
    parser.add_argument(
        "--task-type",
        choices=TASKS.keys(),
        required=True,
        help=f"Type of task. Use one of: {', '.join(TASKS.keys())}",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Either the name of one of the available models: "
            f"{', '.join(MODELS.keys())}; or a Hugging Face model. "
            "The HF model can be either a model available at the HF Hub, or "
            "a model path."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        help=(
            "Pretrained tokenizer name or path (if not the same as `model`). "
            "Must be the same one used for the training of the model to upload."
        ),
    )
    parser.add_argument(
        "--cache-dir", default="./cache", help="Directory for Hugging Face downloaded models."
    )
    parser.add_argument(
        "--experiment-name",
        help=(
            "Suffix of MLFlow experiment. "
            "Must be the same used for the training of the model to upload."
        ),
    )
    parser.add_argument(
        "--run-name", help="Prefix of MLFlow run. Must be the same used for the traning script."
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Token for Hugging Face. If not given will default to $HF_TOKEN env variable.",
    )
    parser.add_argument(
        "--lower-case", action="store_true", help="Should be active for lowercase transformers."
    )
    parser.add_argument(
        "--add-prefix-space", action="store_true", help="Activate for Roberta based tokenizers."
    )
    parser.add_argument(
        "--hf-commit-message", help="Commit message for the upload of the Hugging Face model."
    )
    parser.add_argument(
        "--hf-private-repository",
        action="store_true",
        help="Activate to upload the model as part of a private repository (if it's to be created).",
    )
    parser.add_argument(
        "--hf-revision",
        help=(
            "The revision of the model. It will be stored under a branch with this name and "
            "must be retrieved with that same revision name."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Set for debug mode.")
    config = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if config.debug else logging.INFO,
    )

    if (
        config.model not in MODELS
        and not Path(config.model).is_file()
        and len(list(list_models(search=config.model))) == 0
    ):
        logger.error(
            f"The model {config.model} is not available in the list of models: "
            f"{', '.join(MODELS.keys())}; and is neither a HF file or HF model."
        )
        sys.exit(1)

    if config.tokenizer:
        hf_tokenizer_name_or_path = config.tokenizer
    elif config.model in MODELS:
        hf_tokenizer_name_or_path = MODELS[config.model]
    else:
        hf_tokenizer_name_or_path = config.model

    tokenizer = AutoTokenizer.from_pretrained(
        hf_tokenizer_name_or_path,
        cache_dir=config.cache_dir,
        do_lower_case=config.lower_case,
        use_fast=True,
        add_prefix_space=config.add_prefix_space,
    )

    if config.model in MODELS:
        hf_model_name_or_path = MODELS[config.model]
        model_name = config.model
    else:
        hf_model_name_or_path = config.model
        model_name = (
            os.path.basename(hf_model_name_or_path)
            if os.path.exists(hf_model_name_or_path)
            else hf_model_name_or_path
        )

    mlflow_uri = config.mlflow_dir.absolute().as_uri()
    mlflow_client = mlflow.MlflowClient(mlflow_uri)

    mlflow_train_experiment_name = f"{config.task_type}/{model_name}/train"
    if config.experiment_name:
        mlflow_train_experiment_name += f"/{config.experiment_name}"

    mlflow_train_experiment = mlflow_client.get_experiment_by_name(mlflow_train_experiment_name)
    if mlflow_train_experiment is None:
        logger.error(f"There's no experiment matching the name: {mlflow_train_experiment_name}")
        sys.exit(1)

    mlflow_train_experiment_id = mlflow_train_experiment.experiment_id
    runs = mlflow_client.search_runs(
        experiment_ids=[mlflow_train_experiment_id],
        filter_string=f"run_name LIKE '{config.run_name}%'" if config.run_name else "",
        order_by=["start_time DESC"],
    )
    if not runs:
        logger.error(f"There's no runs for experiment: {mlflow_train_experiment_name}")
        sys.exit(1)

    run = runs[0]
    mlflow_train_experiment_run_id = run.info.run_id
    mlflow_train_experiment_run_name = run.info.run_name
    checkpoint_file = run.data.tags["finalCheckpointPath"]
    if not os.path.exists(checkpoint_file):
        logger.error(f"The checkpoint file {checkpoint_file} doesn't exist.")
        sys.exit(1)
    model = TASKS[config.task_type].load_from_checkpoint(checkpoint_file)

    model.model.push_to_hub(
        repo_id=config.hf_repository,
        token=config.hf_token,
        commit_message=config.hf_commit_message,
        revision=config.hf_revision,
        private=config.hf_private_repository,
    )
    tokenizer.push_to_hub(
        repo_id=config.hf_repository,
        token=config.hf_token,
        commit_message=config.hf_commit_message,
        revision=config.hf_revision,
        private=config.hf_private_repository,
    )

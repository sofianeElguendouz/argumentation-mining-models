"""
Evaluation script for the Argumentation Mining Transformer Module

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
import lightning.pytorch as pl
import mlflow
import os
import pandas as pd
import re
import sys

from datetime import datetime
from huggingface_hub import list_models
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from tempfile import TemporaryDirectory
from typing import Dict, Union

from am_transformer.data import RelationClassificationDataModule, SequenceTaggingDataModule
from am_transformer.models import RelationClassificationTransformerModule, \
    SequenceTaggingTransformerModule
from am_transformer.utils import compute_metrics, compute_seq_tag_labels_metrics


# This is a list of models with an alias, but the script can use other models from Hugging Face
MODELS = {
    'bert': 'bert-base-uncased',
    'deberta-v3': 'microsoft/deberta-v3-base',
    'roberta': 'roberta-base',
    'tiny-bert': 'prajjwal1/bert-tiny'  # Useful for debug purposes
}

# Available tasks to work with
TASKS = {
    'rel-class': (RelationClassificationDataModule, RelationClassificationTransformerModule, 'tsv'),
    'seq-tag': (SequenceTaggingDataModule, SequenceTaggingTransformerModule, 'conll')
}

logger = logging.getLogger(__name__)


def evaluate_model(data_module: pl.LightningDataModule, model: pl.LightningModule,
                   config: argparse.Namespace, trainer: pl.Trainer)\
                       -> Dict[str, Union[float, str]]:
    """
    Evaluates a single model and returns the results as a dictionary. The
    dictionary has different metrics, the classification report and the string
    to store as prediction files (either tsv or conll, depending on the task).

    Parameters
    ==========
    data_module: LightningDataModule
        This is one of the possible Data Modules defined in `TASKS`, either for
        relation classification or for sequence tagging. For more information
        check `acta.data.base.BaseDataModule` and it's children classes.
    model: LightningModule
        The model to be evaluated.
    config: Namespace
        The Namespace configuration that is parsed from the command line via
        argparse.
    trainer: Trainer
        A Pytorch Lightning trainer to run the predictions.

    Returns
    =======
    Dict[str, float | str ]
        The dictionary with the different things that need to be logged.
    """
    decoded_predictions = [
        decoded_prediction
        for batch_prediction in trainer.predict(model=model, datamodule=data_module)
        for decoded_prediction in data_module.decode_predictions(**batch_prediction)
    ]

    if config.task_type == 'rel-class':
        # Predictions have the form (true_label, predicted_label, sentence1, sentence2)
        true_labels = []
        pred_labels = []
        for prediction in decoded_predictions:
            true_labels.append(prediction[0])
            pred_labels.append(prediction[1])
        if config.relevant_labels is not None:
            relevant_labels = config.relevant_labels
        else:
            relevant_labels = [lbl for lbl in data_module.label2id.keys() if lbl != 'noRel']
        metrics = compute_metrics(
            true_labels, pred_labels,
            relevant_labels=relevant_labels,
            prefix="eval"
        )
        metrics["predictions"] = '\n'.join(['\t'.join(pred) for pred in decoded_predictions])
    elif config.task_type == 'seq-tag':
        # Predictions are a list of lists of tuples, where each tuple has the form
        # (token, predicted_label, true_label)
        true_labels = []
        pred_labels = []
        for sentence in decoded_predictions:
            true_labels.extend([token[2] for token in sentence])
            pred_labels.extend([token[1] for token in sentence])
        if config.relevant_labels is not None:
            relevant_labels = config.relevant_labels
        else:
            relevant_labels = [lbl for lbl in data_module.label2id.keys()
                               if lbl not in {'X', 'PAD'}]
        metrics = compute_metrics(
            true_labels, pred_labels,
            relevant_labels=relevant_labels,
            prefix="eval"
        )
        seq_tag_metrics = compute_seq_tag_labels_metrics(
            true_labels, pred_labels,
            labels=list(data_module.label2id.keys()),
            prefix="eval"
        )
        metrics = dict(**metrics, **seq_tag_metrics)
        metrics["predictions"] = '\n\n'.join(['\n'.join(['\t'.join(token) for token in sentence])
                                              for sentence in decoded_predictions])

    sorted_labels = [data_module.id2label[idx] for idx in sorted(data_module.id2label.keys())]
    metrics["classification_report"] = classification_report(
        true_labels, pred_labels,
        # Remove the PAD label as it shouldn't be taken into consideration
        labels=sorted_labels,
        zero_division=0
    )
    metrics["classification_report_relevant"] = classification_report(
        true_labels, pred_labels,
        # Remove the PAD label as it shouldn't be taken into consideration
        labels=[lbl for lbl in sorted_labels if lbl in relevant_labels],
        zero_division=0
    )
    cm = confusion_matrix(
        true_labels, pred_labels, labels=sorted_labels
    )
    metrics["confusion_matrix"] = pd.DataFrame(cm, index=sorted_labels, columns=sorted_labels)

    return metrics


def evaluate_models(data_module: pl.LightningDataModule, config: argparse.Namespace):
    """
    Evaluates the model on the evaluation dataset, calling the `evaluate_model`
    procedure. Depending on the configuration it will evaluate the model from a
    checkpoint or directly from HF hub.

    Parameters
    ==========
    data_module: LightningDataModule
        This is one of the possible Data Modules defined in `TASKS`, either for
        relation classification or for sequence tagging. For more information
        check `acta.data.base.BaseDataModule` and it's children classes.
    config: Namespace
        The Namespace configuration that is parsed from the command line via
        argparse.
    """
    # Setting up the Hugging Face model or path
    if config.model in MODELS:
        hf_model_name_or_path = MODELS[config.model]
        model_name = config.model
    else:
        hf_model_name_or_path = config.model
        model_name = os.path.basename(hf_model_name_or_path)\
            if os.path.exists(hf_model_name_or_path) else hf_model_name_or_path

    # MLFlow Setup
    mlflow_uri = f"file://{config.output_dir.absolute().as_posix()}"

    if not config.eval_without_checkpoint:
        # Try to fetch for a checkpoint to work with
        mlflow_train_experiment_name = f"{config.task_type}/{model_name}/train"
        if config.experiment_name:
            mlflow_train_experiment_name += f"/{config.experiment_name}"
        mlflow_client = mlflow.MlflowClient(mlflow_uri)
        mlflow_train_experiment = mlflow_client.get_experiment_by_name(mlflow_train_experiment_name)
        if mlflow_train_experiment is None:
            logger.error(f"There's no experiment matching the name: {mlflow_train_experiment_name}")
            sys.exit(1)
        mlflow_train_experiment_id = mlflow_train_experiment.experiment_id
        runs = mlflow_client.search_runs(
            experiment_ids=[mlflow_train_experiment_id],
            filter_string=f"run_name LIKE '{config.run_name}%'" if config.run_name else '',
            order_by=['start_time DESC']
        )
        if not runs:
            logger.error(f"There's no runs for experiment: {mlflow_train_experiment_name}")
            sys.exit(1)
        run = runs[0]
        mlflow_train_experiment_run_id = run.info.run_id
        mlflow_train_experiment_run_name = run.info.run_name
        model_or_checkpoint = run.data.tags['finalCheckpointPath']
        if not os.path.exists(model_or_checkpoint):
            logger.error(f"The checkpoint file {model_or_checkpoint} doesn't exist.")
            sys.exit(1)
    else:
        mlflow_train_experiment_name = "N/A"
        mlflow_train_experiment_id = "N/A"
        mlflow_train_experiment_run_id = "N/A"
        mlflow_train_experiment_run_name = "N/A"
        model_or_checkpoint = TASKS[config.task_type][1](
            model_name_or_path=hf_model_name_or_path,
            id2label=data_module.id2label,
            label2id=data_module.label2id,
            config_name_or_path=config.config,
            cache_dir=config.cache_dir,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            classes_weights=data_module.classes_weights if config.weighted_loss else None,
            crf_loss=config.crf_loss
        )

    mlflow_experiment_name = f"{config.task_type}/{model_name}/eval"
    if config.experiment_name:  # Add experiment name as suffix
        mlflow_experiment_name += f"/{config.experiment_name}"
    mlflow_run_name = config.timestamp
    if config.run_name:  # Add the run name as prefix
        mlflow_run_name = f"{config.run_name}/{mlflow_run_name}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_name=mlflow_run_name) as run:
        mlflow.log_params({
            "model_name": model_name,
            "train_experiment_name": mlflow_train_experiment_name,
            "train_experiment_id": mlflow_train_experiment_id,
            "train_experiment_run_id": mlflow_train_experiment_run_id,
            "train_experiment_run_name": mlflow_train_experiment_run_name,
            "random_seed": config.random_seed
        })

        if isinstance(model_or_checkpoint, str):
            # Add a link in the run description pointing to the training experiment run
            run_url = f"#/experiments/{mlflow_train_experiment_id}/"
            run_url += f"runs/{mlflow_train_experiment_run_id}"
            mlflow.set_tag("mlflow.note.content", f"[Training Experiment Run]({run_url})")

        trainer = pl.Trainer(
            accelerator=config.accelerator,
            devices=1,
            precision='16-mixed' if config.fp16 else '32-true',
            logger=False,
            max_epochs=1,
            max_steps=-1,
            limit_test_batches=0.1 if config.debug else 1.0,
            limit_predict_batches=0.1 if config.debug else 1.0,
            num_sanity_val_steps=0
        )

        if isinstance(model_or_checkpoint, str):
            checkpoints_path = os.path.dirname(model_or_checkpoint)
            last_checkpoint = re.search(r"(?<=step=)\d+", model_or_checkpoint)
            if last_checkpoint is None:
                logger.error(f"The checkpoint file '{model_or_checkpoint}' doesn't have a step.")
                sys.exit(1)
            for checkpoint_file in sorted(Path(checkpoints_path).glob("*.ckpt")):
                if not config.eval_all_checkpoints and\
                        checkpoint_file.as_posix() != model_or_checkpoint:
                    # Ignore other checkpoints since we only care about the last one
                    continue
                checkpoint_step = re.search(r"(?<=step=)\d+", checkpoint_file.name)
                if checkpoint_step is None:
                    # Do not run unknown checkpoints when last_checkpoint_step is known
                    logger.warning(f"Ignoring {checkpoint_file} since it doesn't have a step.")
                    continue
                checkpoint_step = int(checkpoint_step.group(0)) if checkpoint_step else None
                model = TASKS[config.task_type][1].load_from_checkpoint(checkpoint_file)
                metrics = evaluate_model(data_module, model, config, trainer)
                classification_report = metrics.pop("classification_report")
                classification_report_relevant = metrics.pop("classification_report_relevant")
                cm = metrics.pop("confusion_matrix")
                predictions = metrics.pop("predictions")
                mlflow.log_metrics(metrics, step=checkpoint_step)
                with TemporaryDirectory() as dh:
                    with open(f"{dh}/report_step={checkpoint_step:05d}.txt", "wt") as fh:
                        print(classification_report, file=fh)
                    mlflow.log_artifact(f"{dh}/report_step={checkpoint_step:05d}.txt")

                    with open(f"{dh}/relevant_report_step={checkpoint_step:05d}.txt", "wt") as fh:
                        print(classification_report_relevant, file=fh)
                    mlflow.log_artifact(f"{dh}/relevant_report_step={checkpoint_step:05d}.txt")

                    with open(f"{dh}/confusion_matrix_step={checkpoint_step:05d}.txt", "wt") as fh:
                        cm.to_string(fh)
                    mlflow.log_artifact(f"{dh}/confusion_matrix_step={checkpoint_step:05d}.txt")

                    predictions_file = f"predictions_step={checkpoint_step:05d}"
                    if config.task_type == "rel-class":
                        predictions_file += ".tsv"
                    else:
                        predictions_file += ".conll"
                    with open(f"{dh}/{predictions_file}", "wt") as fh:
                        print(predictions, file=fh)
                    mlflow.log_artifact(f"{dh}/{predictions_file}")
        else:
            # Evaluate directly on the model
            metrics = evaluate_model(data_module, model_or_checkpoint, config, trainer)
            classification_report = metrics.pop("classification_report")
            classification_report_relevant = metrics.pop("classification_report_relevant")
            cm = metrics.pop("confusion_matrix")
            predictions = metrics.pop("predictions")
            mlflow.log_metrics(metrics)
            with TemporaryDirectory() as dh:
                with open(f"{dh}/report.txt", "wt") as fh:
                    print(classification_report, file=fh)
                mlflow.log_artifact(f"{dh}/report.txt")

                with open(f"{dh}/relevant_report.txt", "wt") as fh:
                    print(classification_report_relevant, file=fh)
                mlflow.log_artifact(f"{dh}/relevant_report.txt")

                with open(f"{dh}/confusion_matrix.txt", "wt") as fh:
                    cm.to_string(fh)
                mlflow.log_artifact(f"{dh}/confusion_matrix.txt")

                predictions_file = "predictions"
                if config.task_type == "rel-class":
                    predictions_file += ".tsv"
                else:
                    predictions_file += ".conll"
                with open(f"{dh}/{predictions_file}", "wt") as fh:
                    print(predictions, file=fh)
                mlflow.log_artifact(f"{dh}/{predictions_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-data",
                        type=Path,
                        required=True,
                        help="The evaluation dataset path. It should already be in the format "
                             "for the corresponding task (`--task-type`).")
    parser.add_argument("--output-dir",
                        required=True,
                        type=Path,
                        help="The output directory where the model predictions will be stored. "
                             "In order to eval trained checkpoint, this should match the output "
                             "directory of the train script.")
    parser.add_argument("--task-type",
                        choices=TASKS.keys(),
                        required=True,
                        help=f"Type of task. Use one of: {', '.join(TASKS.keys())}")
    parser.add_argument("--model",
                        required=True,
                        help="Either the name of one of the available models: "
                             f"{', '.join(MODELS.keys())}; or a Hugging Face model. "
                             "The HF model can be either a model available at the HF Hub, or "
                             "a model path.")
    parser.add_argument("--config",
                        help="Pretrained config name or path (if not the same as `model`).")
    parser.add_argument("--tokenizer",
                        help="Pretrained tokenizer name or path (if not the same as `model`).")
    parser.add_argument("--cache-dir",
                        default="./cache",
                        help="Directory for Hugging Face downloaded models.")
    parser.add_argument("--eval-without-checkpoint",
                        action="store_true",
                        help="If active, it will evaluate the model directly from HF hub.")
    parser.add_argument("--experiment-name",
                        help="Suffix of MLFlow experiment.")
    parser.add_argument("--run-name",
                        help="Prefix of MLFlow run.")
    parser.add_argument("--labels",
                        default=None,
                        nargs="*",
                        help="The list of labels (separated by spaces) for the task. "
                             "If not given it will fallback to the default labels for the task.")
    parser.add_argument("--relevant-labels",
                        default=None,
                        nargs="*",
                        help="The list of relevant labels for the task, so it will calculate "
                             "the metrics with these relevant labels in consideration. If not "
                             "given it will fall back to the relevant labels for the task.")
    parser.add_argument("--accelerator",
                        default="auto",
                        help="What device to use as accelerator (cpu, gpu, tpu, etc).")
    parser.add_argument("--num-workers",
                        default=-1,
                        type=int,
                        help="Number of workers to use for DataLoaders. Set to -1 to use all cpus.")
    parser.add_argument("--batch-size",
                        default=32,
                        type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--max-seq-length",
                        type=int,
                        help="The maximum total input sequence length after tokenization."
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded. "
                             "If left empty it will truncate to the model's max size and pad to "
                             "the maximum size of each training step.")
    parser.add_argument("--lower-case",
                        action="store_true",
                        help="Should be active for lowercase transformers.")
    parser.add_argument("--eval-all-checkpoints",
                        action="store_true",
                        help="Evaluate all checkpoints for the model.")
    parser.add_argument("--random-seed",
                        default=42,
                        type=int,
                        help="Initial random seed.")
    parser.add_argument("--weighted-loss",
                        action="store_true",
                        help="Only useful for Relationship Classification trainings. "
                             "If true the loss function is weighted inversely by class.")
    parser.add_argument("--crf-loss",
                        action="store_true",
                        help="Only useful for Sequence Tagging trainings. "
                             "If true the loss function uses Conditional Random Fields.")
    parser.add_argument("--fp16",
                        action="store_true",
                        help="Whether to use 16-bit (mixed) precision")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Set for debug mode.")
    config = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if config.debug else logging.INFO,
    )

    if config.eval_without_checkpoint and config.eval_all_checkpoints:
        logger.error("Incompatible options present. Either choose `--eval-without-checkpoint`, "
                     "or `--eval-all-checkpoints`.")
        sys.exit(1)

    if config.model not in MODELS and not Path(config.model).is_file() \
            and len(list(list_models(search=config.model))) == 0:
        logger.error(f"The model {config.model} is not available in the list of models: "
                     f"{', '.join(MODELS.keys())}; and is neither a HF file or HF model.")
        sys.exit(1)

    data_splits = {
        'test': config.test_data
    }

    logger.info(f"Accelerator: {config.accelerator}.")

    # Timestamp to keep track of results
    config.timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Set random seed
    pl.seed_everything(config.random_seed)

    if config.tokenizer:
        hf_tokenizer_name_or_path = config.tokenizer
    elif config.model in MODELS:
        hf_tokenizer_name_or_path = MODELS[config.model]
    else:
        hf_tokenizer_name_or_path = config.model

    # Instantiate data module
    data_module = TASKS[config.task_type][0](
        data_splits=data_splits,
        tokenizer_name_or_path=hf_tokenizer_name_or_path,
        labels=config.labels,
        tokenizer_config=dict(
            cache_dir=config.cache_dir,
            do_lower_case=config.lower_case,
            use_fast=True,
            add_prefix_space=True if hf_tokenizer_name_or_path == 'roberta-base' else False
        ),
        datasets_config=dict(
            max_seq_length=config.max_seq_length
        ),
        eval_batch_size=config.batch_size,
        evaluation_split='test',
        num_workers=config.num_workers
    )
    data_module.prepare_data()
    data_module.setup('fit')

    evaluate_models(data_module, config)

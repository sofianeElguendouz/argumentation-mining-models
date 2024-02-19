#!/usr/bin/env python
"""
Trainer script for the Argumentation Mining Transformer Module

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
import sys

from datetime import datetime
from huggingface_hub import list_models
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from pathlib import Path

from amtm.data import RelationClassificationDataModule, SequenceTaggingDataModule
from amtm.models import RelationClassificationTransformerModule, \
    SequenceTaggingTransformerModule


# This is a list of models with an alias, but the script can use other models from Hugging Face
MODELS = {
    'bert': 'bert-base-uncased',
    'deberta-v3': 'microsoft/deberta-v3-base',
    'roberta': 'roberta-base',
    'tiny-bert': 'prajjwal1/bert-tiny'  # Useful for debug purposes
}

# Available tasks to work with
TASKS = {
    'rel-class': (RelationClassificationDataModule, RelationClassificationTransformerModule),
    'seq-tag': (SequenceTaggingDataModule, SequenceTaggingTransformerModule)
}

logger = logging.getLogger(__name__)


def train_model(data_module: pl.LightningDataModule, model: pl.LightningModule,
                config: argparse.Namespace):
    """
    Trains a model.

    Parameters
    ==========
    data_module: LightningDataModule
        This is one of the possible Data Modules defined in `TASKS`, either for
        relation classification or for sequence tagging. For more information
        check `amtm.data.base.BaseDataModule` and it's children classes.
    model: LightningModule
        This is one of the possible Lightning Modules in `TASKS`, either for
        relation classification or for sequence tagging. For more information
        check `amtm.models.base.BaseTransformerModule` and it's children classes.
    config: Namespace
        The Namespace configuration that is parsed from the command line via
        argparse.
    """
    if config.model in MODELS:
        model_name = config.model
    elif os.path.exists(config.model):
        model_name = os.path.basename(config.model)
    else:
        model_name = config.model

    # MLFlow Setup
    mlflow_uri = config.output_dir.absolute().as_uri()
    mlflow_experiment_name = f"{config.task_type}/{model_name}/train"
    if config.experiment_name:  # Add experiment name as suffix
        mlflow_experiment_name += f"/{config.experiment_name}"
    mlflow_run_name = config.timestamp
    if config.run_name:  # Add the run name as prefix
        mlflow_run_name = f"{config.run_name}/{mlflow_run_name}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_name=mlflow_run_name) as run:
        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_experiment_name,
            run_name=mlflow_run_name,
            tracking_uri=mlflow_uri,
            log_model=False,
            run_id=run.info.run_id
        )

        mlflow.log_params({
            "model_name": model_name,
            "model_checkpoint": config.load_from_checkpoint or "N/A",
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "early_stopping": config.early_stopping if config.validation else "N/A",
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_grad_norm": config.max_grad_norm,
            "random_seed": config.random_seed,
            "weighted_loss": config.weighted_loss
        })

        callbacks = []
        # FIXME: This is a little hackish, but good for now
        checkpoint_path = Path(run.info.artifact_uri.removeprefix("file://")) / "checkpoints"
        model_checkpoints = ModelCheckpoint(
            dirpath=checkpoint_path,
            filename=mlflow_experiment_name.replace("/", "_") + "_{epoch:02d}_{step:05d}",
            save_top_k=-1,  # Save all models
            every_n_train_steps=config.save_every_n_steps,
            enable_version_counter=False  # Overwrite existing checkpoints
        )
        callbacks.append(model_checkpoints)

        if config.early_stopping and config.validation:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                min_delta=1e-6,
                patience=config.early_stopping
            )
            callbacks.append(early_stopping)

        trainer = pl.Trainer(
            accelerator=config.accelerator,
            devices=config.num_devices,
            strategy='ddp_find_unused_parameters_true',
            precision='16-mixed' if config.fp16 else '32-true',
            logger=mlflow_logger,
            callbacks=callbacks,
            max_epochs=config.epochs,
            val_check_interval=config.log_every_n_steps,
            accumulate_grad_batches=config.gradient_accumulation_steps,
            gradient_clip_val=config.max_grad_norm if config.max_grad_norm else None,
            limit_train_batches=0.1 if config.debug else 1.0,  # Use only 10% of training for debug
            limit_test_batches=0.1 if config.debug else 1.0,
            limit_predict_batches=0.1 if config.debug else 1.0,
            limit_val_batches=0 if not config.validation else 0.1 if config.debug else 1.0,
            num_sanity_val_steps=0 if not config.validation else 1 if config.debug else 2
        )

        logger.info("Starting model training routine")
        trainer.fit(model, datamodule=data_module, ckpt_path=config.load_from_checkpoint)
        logger.info("Finished model training routine")

        logger.info("Saving last model checkpoint")
        last_model_checkpoint = Path(model_checkpoints.format_checkpoint_name(
            {"epoch": trainer.current_epoch, "step": trainer.global_step}
        ))
        # Save a checkpoint for the last epoch and last step
        trainer.save_checkpoint(last_model_checkpoint)

        if trainer.is_global_zero:
            # Only in the main process, record the path to the last checkpoint as a MLFlow tag
            mlflow.set_tag("finalCheckpointPath", last_model_checkpoint.absolute().as_posix())

    # After the experiment is finished, we need to run a cleanup on MLFlow runs
    # that were created by DDP strategy (that spawned child processes), this runs
    # don't store any particularly useful information
    if not trainer.is_global_zero:
        logger.info(f"Cleaning up extra run: {run.info.run_id}")
        mlflow.MlflowClient(mlflow_uri).delete_run(run.info.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data",
                        type=Path,
                        required=True,
                        help="The train dataset path. It should already be in the format "
                             "for the corresponding task (`--task-type`).")
    parser.add_argument("--validation-data",
                        type=Path,
                        help="The validation dataset path. It should already be in the format "
                             "for the corresponding task (`--task-type`).")
    parser.add_argument("--output-dir",
                        required=True,
                        type=Path,
                        help="The directory where the model logs and checkpoints will be stored.")
    parser.add_argument("--task-type",
                        choices=TASKS.keys(),
                        required=True,
                        help=f"Type of task. Use one of: {', '.join(TASKS.keys())}")
    parser.add_argument("--model",
                        required=True,
                        help="Either the name of one of the available models: "
                             f"{', '.join(MODELS.keys())}; or a Hugging Face model. "
                             "The HF model can be either a model available at the HF Hub, or "
                             "a model path. To load a checkpoint reached using this same trainer "
                             "script please use the `--load-from-checkpoint` option.")
    parser.add_argument("--config",
                        help="Pretrained config name or path (if not the same as `model`).")
    parser.add_argument("--tokenizer",
                        help="Pretrained tokenizer name or path (if not the same as `model`).")
    parser.add_argument("--cache-dir",
                        default="./cache",
                        help="Directory for Hugging Face downloaded models.")
    parser.add_argument("--load-from-checkpoint",
                        help="Path to a checkpoint file to continue training.")
    parser.add_argument("--experiment-name",
                        help="Suffix of MLFlow experiment.")
    parser.add_argument("--run-name",
                        help="Prefix of MLFlow run.")
    parser.add_argument("--labels",
                        default=None,
                        nargs="*",
                        help="The list of labels (separated by spaces) for the task. "
                             "If not given it will fallback to the default labels for the task.")
    parser.add_argument("--accelerator",
                        default="auto",
                        help="What device to use as accelerator (cpu, gpu, tpu, etc).")
    parser.add_argument("--num-devices",
                        default=-1,
                        type=int,
                        help="Number of devices to use. If not given selects automatically.")
    parser.add_argument("--num-workers",
                        default=-1,
                        type=int,
                        help="Number of workers to use for DataLoaders. Set to -1 to use all cpus.")
    parser.add_argument("--epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--early-stopping",
                        default=2,
                        type=int,
                        help="If > 0 then stops if there are `early-stopping` logs without "
                             "improvement on the validation loss.")
    parser.add_argument("--batch-size",
                        default=8,
                        type=int,
                        help="Batch size (per GPU/CPU) for training.")
    parser.add_argument("--gradient-accumulation-steps",
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before "
                             "performing a backward/update pass.")
    parser.add_argument("--max-grad-norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm (for gradient clipping). Set to 0 to deactivate.")
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
    parser.add_argument("--learning-rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for AdamW.")
    parser.add_argument("--weight-decay",
                        default=0.0,
                        type=float,
                        help="Weight decay for parameters that accept it.")
    parser.add_argument("--warmup-steps",
                        default=0,
                        type=int,
                        help="Number of steps for linear warmup.")
    parser.add_argument("--weighted-loss",
                        action="store_true",
                        help="Only useful for Relationship Classification trainings. "
                             "If true the loss function is weighted inversely by class.")
    parser.add_argument("--crf-loss",
                        action="store_true",
                        help="Only useful for Sequence Tagging trainings. "
                             "If true the loss function uses Conditional Random Fields.")
    parser.add_argument("--log-every-n-steps",
                        default=50,
                        type=int,
                        help="Log every N update steps.")
    parser.add_argument("--save-every-n-steps",
                        default=50,
                        type=int,
                        help="Save checkpoint every N update steps.")
    parser.add_argument("--random-seed",
                        default=42,
                        type=int,
                        help="Initial random seed.")
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

    if config.model not in MODELS and not Path(config.model).is_file() \
            and len(list(list_models(search=config.model))) == 0:
        logger.error(f"The model {config.model} is not available in the list of models: "
                     f"{', '.join(MODELS.keys())}; and is neither a HF file or HF model.")
        sys.exit(1)

    data_splits = {
        'train': config.train_data
    }
    if config.validation_data is not None:
        data_splits['validation'] = config.validation_data
        config.validation = True
    else:
        config.validation = False

    config.num_devices = config.num_devices if config.num_devices > 0 else "auto"

    logger.info(
        f"Accelerator: {config.accelerator}. - "
        f"No. of devices: {config.num_devices}. -"
        f"16-bit precision training: {config.fp16}."
    )

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
        train_batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    data_module.prepare_data()
    data_module.setup('fit')

    # Setting up the Hugging Face model or path
    if config.model in MODELS:
        hf_model_name_or_path = MODELS[config.model]
    else:
        hf_model_name_or_path = config.model

    # Instantiate (or load) model
    if config.load_from_checkpoint is not None:
        if not Path(config.load_from_checkpoint).is_file():
            logger.error(f"The checkpoint file doesn't exists: {config.load_from_checkpoint}")
            sys.exit(1)
        model = TASKS[config.task_type][1].load_from_checkpoint(config.load_from_checkpoint)
    else:
        model = TASKS[config.task_type][1](
            model_name_or_path=hf_model_name_or_path,
            id2label=data_module.id2label,
            label2id=data_module.label2id,
            config_name_or_path=config.config,
            cache_dir=config.cache_dir,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            classes_weights=data_module.classes_weights if config.weighted_loss else None,
            crf_loss=config.crf_loss
        )

    train_model(data_module, model, config)

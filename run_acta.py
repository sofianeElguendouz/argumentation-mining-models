"""
Trainer script for the ACTA modules.

   Copyright 2023 The ANTIDOTE Project Contributors <https://univ-cotedazur.eu/antidote>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import argparse
import logging
import lightning.pytorch as pl
import os
import sys

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path, PosixPath
from typing import Dict

from acta.data import RelationClassificationDataModule, SequenceTaggingDataModule
from acta.models import RelationClassificationTransformerModule, SequenceTaggingTransformerModule
from acta.utils import TTYAwareProgressBar


MODELS = {
    'bert': 'bert-base-uncased',
    'biobert': 'monologg/biobert_v1.1_pubmed',
    'deberta-v3': 'microsoft/deberta-v3-base',
    'roberta': 'roberta-base',
    'scibert-monolog': 'monologg/scibert_scivocab_uncased',
    'scibert-allenai': 'allenai/scibert_scivocab_uncased',
    'xlm-roberta': 'xlm-roberta-base'
}
TASKS = {
    'rel-class': (RelationClassificationDataModule, RelationClassificationTransformerModule, 'tsv'),
    'seq-tag': (SequenceTaggingDataModule, SequenceTaggingTransformerModule, 'conll')
}
DATA_SPLITS = ['train', 'test', 'validation', 'dev']

logger = logging.getLogger(__name__)


def get_data_splits(input_dir: str, task_type: str) -> Dict[str, PosixPath]:
    """
    Function that search for possible split files given an input directory.

    It will return a map between each of the three possible splits (train, test,
    validation) and the path to the corresponding file. Since there could be a
    `dev.*` file that is usually used for validation purposes, the function
    maps that file to the `validation` split in order to keep the convention used
    in Lightning and HuggingFace.

    It will search the files based on the task name.

    Parameters
    ----------
    input_dir: str
        Path to the input dir with the split files.
    task_type: str
        Used to determine the file extension.

    Returns
    -------
    Dict[str, PosixPath]
        A mapping between the split name and the associated file path to that
        split.
    """
    assert task_type in TASKS, f"Invalid task, use on of: {', '.join(TASKS.keys())}."
    input_dir = Path(input_dir)
    file_type = TASKS[task_type][2]
    splits_files = {}
    for split in DATA_SPLITS:
        split_file = list(input_dir.glob(f"{split}*.{file_type}"))
        if split_file:
            if split == 'dev' and 'validation' in splits_files:
                logger.warning(
                    f"Ignoring the file {split_file[0]} since there is {splits_files['validation']}"
                )
            elif split == 'validation' and 'validation' in splits_files:
                # It detected the existence of a `dev.*` file and it uses that
                logger.warning(
                    f"Ignoring the file {split_file[0]} since there is {splits_files['validation']}"
                )
            elif len(split_file) > 1:
                logger.warning(
                    f"There's more than 1 file for {split} split. Using {split_file[0]}."
                )
            # We use the `dev.*` files as validation files and rename the split to match the
            # convention both for Lightning and HuggingFace
            split = 'validation' if split == 'dev' else split
            splits_files[split] = split_file[0]
    return splits_files


def train_model(data_module: pl.LightningDataModule, model: pl.LightningModule,
                args: argparse.Namespace) -> ModelCheckpoint:
    """
    Trains a model and returns the checkpoints for that model.
    """
    output_dir = Path(args.output_dir)
    model_name = args.model if args.model in MODELS else os.path.basename(args.model)
    callbacks = []

    model_logger = TensorBoardLogger(save_dir=output_dir / args.logging_dir)

    model_checkpoints = ModelCheckpoint(
        dirpath=output_dir / args.checkpoint_path,
        filename=model_name + "-{epoch:02d}-{step:05d}",
        save_top_k=-1,  # Save all models
        every_n_train_steps=args.save_every_n_steps
    )
    callbacks.append(model_checkpoints)

    progress_bar = TTYAwareProgressBar(refresh_rate=args.log_every_n_steps)
    callbacks.append(progress_bar)

    if args.early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-6,
            patience=args.early_stopping
        )
        callbacks.append(early_stopping)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.num_devices,
        precision='16-mixed' if args.fp16 else '32-true',
        logger=model_logger,
        callbacks=callbacks,
        max_epochs=args.epochs if args.max_steps < 0 else None,
        max_steps=args.max_steps,
        val_check_interval=args.log_every_n_steps,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_grad_norm if args.max_grad_norm else None,
        limit_train_batches=0.1 if args.debug else 1.0,  # Use only 10% of training for debug
        limit_test_batches=0.5 if args.debug else 1.0,
        limit_val_batches=0 if not args.validation else 1.0,
        num_sanity_val_steps=0 if not args.validation else 2
    )

    logger.info("Starting model training routine")
    trainer.fit(model, datamodule=data_module)
    logger.info("Finished model training routine")

    logger.info("Saving last model checkpoint")
    trainer.save_checkpoint(output_dir / args.checkpoint_path / (model_name + "-final.ckpt"))

    return model_checkpoints


def evaluate_models(data_module: pl.LightningDataModule, model: pl.LightningModule,
                    model_checkpoints: ModelCheckpoint, args: argparse.Namespace):
    """
    Evaluates the model on the evaluation dataset. Depending on the options, it
    will evaluate only in the final model or in all the models in the
    checkpoints.
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input-dir",
                        required=True,
                        help="The input directory. It has the train, test and validation (dev) "
                             "files. Depending on the task they might be tsv or conll.")
    parser.add_argument("--output-dir",
                        required=True,
                        help="The output directory where the model predictions and checkpoints "
                             "will be stored.")
    parser.add_argument("--task-type",
                        choices=TASKS.keys(),
                        required=True,
                        help=f"Type of task. Use one of: {', '.join(TASKS.keys())}")
    parser.add_argument("--model",
                        required=True,
                        help="Either the name of one of the available models: "
                             f"{', '.join(MODELS.keys())}; or a path to a pre-trained model file. "
                             "If using a file, it must be a Hugging Face model path. To load a "
                             "checkpoint reached using this same trainer script please use the "
                             "`--load-from-checkpoint` option.")

    # Other parameters
    parser.add_argument("--config",
                        help="Pretrained config name or path (if not the same as `model`).")
    parser.add_argument("--tokenizer",
                        help="Pretrained tokenizer name or path (if not the same as `model`).")
    parser.add_argument("--cache-dir",
                        help="Directory for Hugging Face downloaded models.")
    parser.add_argument("--checkpoint-path",
                        default="checkpoints",
                        help="Name of directory (inside output-dir) to store the checkpoint files.")
    parser.add_argument("--load-from-checkpoint",
                        help="Path to a checkpoint file to continue training.")
    parser.add_argument("--logging-dir",
                        default="logs",
                        help="Name of directory (inside output-dir) to store the TensorBoard logs.")
    parser.add_argument("--train",
                        action="store_true",
                        help="Train the model.")
    parser.add_argument("--evaluation-split",
                        choices=["train", "test", "validation"],
                        help="The split to use for evaluation at the end of training "
                             "(train, validation, test)."
                             "If not given there won't be any evaluation done.")
    parser.add_argument("--validation",
                        action="store_true",
                        help="If active, runs validation after `--log-every-n-steps` steps. "
                             "Validation is useful for early stopping of the training process.")
    parser.add_argument("--accelerator",
                        default="auto",
                        help="What device to use as accelerator (cpu, gpu, tpu, etc).")
    parser.add_argument("--num-devices",
                        default="auto",
                        type=int,
                        help="Number of devices to use. If not given selects automatically.")
    parser.add_argument("--epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--early-stopping",
                        default=0,
                        type=int,
                        help="If > 0 then stops if there are `early-stopping` logs without "
                             "improvement on the validation loss.")
    parser.add_argument("--max-steps",
                        default=-1,
                        type=int,
                        help="If > 0: set total number of training steps to perform. "
                             "Overrides epochs.")
    parser.add_argument("--train-batch-size",
                        default=8,
                        type=int,
                        help="Batch size (per GPU/CPU) for training.")
    parser.add_argument("--eval-batch-size",
                        default=8,
                        type=int,
                        help="Batch size (per GPU/CPU) for evaluation.")
    parser.add_argument("--gradient-accumulation-steps",
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before "
                             "performing a backward/update pass.")
    parser.add_argument("--max-grad-norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm (for gradient clipping). Set to 0 to deactivate.")
    parser.add_argument("--learning-rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for AdamW.")
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
    parser.add_argument("--weight-decay",
                        default=0.0,
                        type=float,
                        help="Weight decay for parameters that accept it.")
    parser.add_argument("--adam-epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup-steps",
                        default=0,
                        type=int,
                        help="Number of steps for linear warmup.")
    parser.add_argument("--log-every-n-steps",
                        default=50,
                        type=int,
                        help="Log every N update steps.")
    parser.add_argument("--save-every-n-steps",
                        default=50,
                        type=int,
                        help="Save checkpoint every N update steps.")
    parser.add_argument("--eval-all-checkpoints",
                        action="store_true",
                        help="Evaluate all checkpoints for the model.")
    parser.add_argument("--overwrite-output",
                        action="store_true",
                        help="Overwrite the content of the output directory.")
    parser.add_argument("--random-seed",
                        default=42,
                        type=int,
                        help="Initial random seed.")
    parser.add_argument("--fp16",
                        action="store_true",
                        help="Whether to use 16-bit (mixed) precision")
    parser.add_argument("--server_ip",
                        default="",
                        help="For distant debugging.")
    parser.add_argument("--server_port",
                        default="",
                        help="For distant debugging.")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Set for debug mode.")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    # Checking pre-conditions
    if not args.train and not args.evaluation_split:
        logger.error("The script must be for training or at least have 1 evaluation split.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and list(output_dir.glob('*')) and args.train\
            and not args.overwrite_output:
        logger.error(f"Output directory ({args.output_dir}) already exists and is not empty. "
                     "Use --overwrite-output to ovewrite the directory (information will be lost).")
        sys.exit()

    if args.model not in MODELS and not Path(args.model).is_file():
        logger.error(f"The model {args.model} is not available in the list of models: "
                     f"{', '.join(MODELS.keys())}; and is not an existing file.")
        sys.exit()

    data_splits = get_data_splits(args.input_dir, args.task_type)
    if not data_splits:
        logger.error("There are no files to train nor evaluate. Exiting the trainer.")
        sys.exit(1)

    if args.train and 'train' not in data_splits:
        logger.error("There's no file for training.")
        sys.exit(1)

    if args.evaluation_split and args.evaluation_split not in data_splits:
        logger.error(f"The evaluation split {args.evaluation_split} file is missing.")
        sys.exit(1)

    if args.validation and 'validation' not in data_splits:
        logger.error("There's no file for validation.")
        sys.exit(1)

    if args.early_stopping and 'validation' not in data_splits:
        logger.error("There's no validation file for early stopping")
        sys.exit(1)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging
        # see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        logger.debug("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    logger.info(
        f"Accelerator: {args.accelerator}.\n"
        f"No. of devices: {args.num_devices}.\n"
        f"16-bit precision training: {args.fp16}."
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed
    pl.seed_everything(args.random_seed)

    data_module = TASKS[args.task][0](
        data_splits=data_splits,
        tokenizer_name_or_path=args.tokenizer if args.tokenizer else args.model,
        tokenizer_config=dict(
            cache_dir=args.cache_dir,
            do_lower_case=args.lower_case,
            use_fast=True
        ),
        datasets_config=dict(
            max_seq_length=args.max_seq_length
        ),
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        evaluation_split=args.evaluation_split
    )
    data_module.prepare_data()
    data_module.setup('fit')

    if Path(args.start_from_checkpoint).is_file():
        model = TASKS[args.task][1].load_from_checkpoint(args.load_from_checkpoint)
    else:
        model = TASKS[args.task][1](
            model_name_or_path=args.model,
            id2label=data_module.id2label,
            label2id=data_module.label2id,
            config_name_or_path=args.config,
            cache_dir=args.cache_dir,
            masked_label=data_module.label2id.get('PAD', -100),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            adam_epsilon=args.adam_epsilon,
            warmup_steps=args.warmup_steps
        )

    if args.train:
        model_checkpoints = train_model(data_module, model, args)

    if args.evaluation_split:
        evaluate_models(data_module, model, model_checkpoints, args)

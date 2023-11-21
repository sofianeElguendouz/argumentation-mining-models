ANTIDOTE ACTA MODULE
====================

This repository has a re-implementation of the *Transformed-based Argument
Mining* module presented in the work by T. Mayer, E. Cabrio and S. Villata:
[Transformer-based Argument Mining for Healthcare
Applications](https://hal.archives-ouvertes.fr/hal-02879293) (ECAI 2020)

This repository is part of the ANTIDOTE Project on Explainable AI. For more
information please check:
https://www.inria.fr/en/explainable-ai-algorithm-learning.

The ACTA Module is the basis for the [**A**rgumentative **C**linical **T**rial
**A**nalysis tool](http://ns.inria.fr/acta/).

It has implementations for training two kinds of tasks:

1. Argument Component Detection (Sequence Tagging Task), and
2. Argument Relation Classification (Sequence Classification Task)

This repository was forked from the [original work by Tobias
Mayer](https://gitlab.com/tomaye/ecai2020-transformer_based_am). His original
code is available at [`./archive`](./archive). We updated it so it was able to
handle the latests [Hugging Face models](https://huggingface.co/models)
available.

Requirements
------------

- The code was tested under Python 3.10.8. It might run under lower versions of
  Python but it wasn't tested on those.
- The required packages are listed in [`requirements.txt`](./requirements.txt).
  - Alternatively you have a Conda Environment in
    [`environment.yml`](./environment.yml). But bear in mind this was done for
    internal development of the module to run it on our hardware, its not
    guaranteed to run in yours.
- The code is heavily dependent on the following libraries:
  - [Lightning](https://lightning.ai/) >= 2: Developed with 2.0.9
  - [Hugging Face](https://huggingface.co/) >= 4: Developed with 4.33.2
  - [Pytorch-CRF](https://pytorch-crf.readthedocs.io/en/stable/): Developed with
    0.7.2

Usage
-----

We provide with the [`run_acta.py`](./run_acta.py) script for training and
evaluating models. You can check the available options running:

    python run_acta.py --help

Here is the list of options:

    usage: run_acta.py [-h] [--input-dir INPUT_DIR] [--train-data TRAIN_DATA]
                    [--test-data TEST_DATA] [--validation-data VALIDATION_DATA]
                    --output-dir OUTPUT_DIR --task-type {rel-class,seq-tag}
                    --model MODEL [--config CONFIG] [--tokenizer TOKENIZER]
                    [--cache-dir CACHE_DIR] [--checkpoint-path CHECKPOINT_PATH]
                    [--timestamp-checkpoints]
                    [--load-from-checkpoint LOAD_FROM_CHECKPOINT]
                    [--logging-dir LOGGING_DIR] [--train]
                    [--evaluation-split {train,test,validation}] [--validation]
                    [--labels [LABELS ...]]
                    [--relevant-labels [RELEVANT_LABELS ...]]
                    [--accelerator ACCELERATOR] [--num-devices NUM_DEVICES]
                    [--num-workers NUM_WORKERS] [--epochs EPOCHS]
                    [--early-stopping EARLY_STOPPING] [--max-steps MAX_STEPS]
                    [--train-batch-size TRAIN_BATCH_SIZE]
                    [--eval-batch-size EVAL_BATCH_SIZE]
                    [--gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS]
                    [--max-grad-norm MAX_GRAD_NORM]
                    [--max-seq-length MAX_SEQ_LENGTH] [--lower-case]
                    [--learning-rate LEARNING_RATE]
                    [--weight-decay WEIGHT_DECAY] [--adam-epsilon ADAM_EPSILON]
                    [--warmup-steps WARMUP_STEPS]
                    [--log-every-n-steps LOG_EVERY_N_STEPS]
                    [--save-every-n-steps SAVE_EVERY_N_STEPS]
                    [--eval-all-checkpoints] [--overwrite-output]
                    [--random-seed RANDOM_SEED] [--fp16] [--debug]

    options:
    -h, --help            show this help message and exit
    --input-dir INPUT_DIR
                            The input directory. It has the train, test and
                            validation (dev) files. Depending on the task they
                            might be tsv or conll. If given, the parameters
                            `--train-data`, `--test-data` and `--validation-data`
                            will be ignored.
    --train-data TRAIN_DATA
                            The train dataset path. It should already be in the
                            format for the corresponding task (`--task-type`).
    --test-data TEST_DATA
                            The test dataset path. It should already be in the
                            format for the corresponding task (`--task-type`).
    --validation-data VALIDATION_DATA
                            The validation dataset path. It should already be in
                            the format for the corresponding task (`--task-type`).
    --output-dir OUTPUT_DIR
                            The output directory where the model predictions and
                            checkpoints will be stored.
    --task-type {rel-class,seq-tag}
                            Type of task. Use one of: rel-class, seq-tag
    --model MODEL         Either the name of one of the available models: bert,
                            biobert, deberta-v3, roberta, scibert-monolog,
                            scibert-allenai, xlm-roberta, tiny-bert; or a path to
                            a pre-trained model file. If using a file, it must be
                            a Hugging Face model path. To load a checkpoint
                            reached using this same trainer script please use the
                            `--load-from-checkpoint` option.
    --config CONFIG       Pretrained config name or path (if not the same as
                            `model`).
    --tokenizer TOKENIZER
                            Pretrained tokenizer name or path (if not the same as
                            `model`).
    --cache-dir CACHE_DIR
                            Directory for Hugging Face downloaded models.
    --checkpoint-path CHECKPOINT_PATH
                            Name of directory (inside output-dir) to store the
                            checkpoint files.
    --timestamp-checkpoints
                            If active, it will create a directory under
                            `--checkpoint-path` with a timestamp to store the
                            checkpoints (this is to avoid checkpoint overwriting
                            when running multiple experiments in parallel).
                            Warning: This is only valid in training mode. If you
                            want to evaluate existing checkpoint inside a
                            directory with timestamp you need to provide the path
                            to the checkpoints directory in full.
    --load-from-checkpoint LOAD_FROM_CHECKPOINT
                            Path to a checkpoint file to continue training.
    --logging-dir LOGGING_DIR
                            Name of directory (inside output-dir) to store the
                            TensorBoard logs.
    --train               Train the model.
    --evaluation-split {train,test,validation}
                            The split to use for evaluation at the end of training
                            (train, validation, test).If not given there won't be
                            any evaluation done.
    --validation          If active, runs validation after `--log-every-n-steps`
                            steps. Validation is useful for early stopping of the
                            training process.
    --labels [LABELS ...]
                            The list of labels (separated by spaces) for the task.
                            If not given it will fallback to the default labels
                            for the task.
    --relevant-labels [RELEVANT_LABELS ...]
                            The list of relevant labels for the task, so it will
                            calculate the metrics with these relevant labels in
                            consideration. If not given it will fall back to the
                            relevant labels for the task.
    --accelerator ACCELERATOR
                            What device to use as accelerator (cpu, gpu, tpu,
                            etc).
    --num-devices NUM_DEVICES
                            Number of devices to use. If not given selects
                            automatically.
    --num-workers NUM_WORKERS
                            Number of workers to use for DataLoaders. Set to -1 to
                            use all cpus.
    --epochs EPOCHS       Total number of training epochs to perform.
    --early-stopping EARLY_STOPPING
                            If > 0 then stops if there are `early-stopping` logs
                            without improvement on the validation loss.
    --max-steps MAX_STEPS
                            If > 0: set total number of training steps to perform.
                            Overrides epochs.
    --train-batch-size TRAIN_BATCH_SIZE
                            Batch size (per GPU/CPU) for training.
    --eval-batch-size EVAL_BATCH_SIZE
                            Batch size (per GPU/CPU) for evaluation.
    --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS
                            Number of updates steps to accumulate before
                            performing a backward/update pass.
    --max-grad-norm MAX_GRAD_NORM
                            Max gradient norm (for gradient clipping). Set to 0 to
                            deactivate.
    --max-seq-length MAX_SEQ_LENGTH
                            The maximum total input sequence length after
                            tokenization.Sequences longer than this will be
                            truncated, sequences shorter will be padded. If left
                            empty it will truncate to the model's max size and pad
                            to the maximum size of each training step.
    --lower-case          Should be active for lowercase transformers.
    --learning-rate LEARNING_RATE
                            The initial learning rate for AdamW.
    --weight-decay WEIGHT_DECAY
                            Weight decay for parameters that accept it.
    --adam-epsilon ADAM_EPSILON
                            Epsilon for Adam optimizer.
    --warmup-steps WARMUP_STEPS
                            Number of steps for linear warmup.
    --log-every-n-steps LOG_EVERY_N_STEPS
                            Log every N update steps.
    --save-every-n-steps SAVE_EVERY_N_STEPS
                            Save checkpoint every N update steps.
    --eval-all-checkpoints
                            Evaluate all checkpoints for the model.
    --overwrite-output    Overwrite the content of the output directory.
    --random-seed RANDOM_SEED
                            Initial random seed.
    --fp16                Whether to use 16-bit (mixed) precision
    --debug               Set for debug mode.

We have a sample bash script with some common configurations in
[`run_acta.sh`](./run_acta.sh) that you can check.
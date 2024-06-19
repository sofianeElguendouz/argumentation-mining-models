Argumentation Mining Transformers Module (AMTM)
===============================================

This repository has an implementation of the *Transformed-based Argument Mining*
work presented by T. Mayer, E. Cabrio and S. Villata:
[Transformer-based Argument Mining for Healthcare Applications](https://hal.archives-ouvertes.fr/hal-02879293)
(ECAI 2020)

It was originally forked from the ANTIDOTE ACTA Module, which is part of the
ANTIDOTE Project on Explainable AI. For more information check:
https://gitlab.com/wimmics-antidote/antidote-acta/

The module has the implementation of two tasks for Argumentation Mining:

1. Argument Component Detection (Sequence Tagging Task), and
2. Argument Relation Classification (Sequence Classification Task)

Requirements
------------

- The code was tested under Python 3.10.8. It might run for previous versions of
  Python but it wasn't tested on those.
- The required packages are listed in [`requirements.txt`](./requirements.txt).
- The code is heavily dependent on the following libraries:
  - [PyTorch](https://pytorch.org) >= 2: Developed with 2.2.0
  - [Lightning](https://lightning.ai/) >= 2: Developed with 2.1.3
  - [Hugging Face](https://huggingface.co/) >= 4: Developed with 4.33.2
- For running the training and evaluation scripts you also need to install the
  development packages:
  - These are listed in [`dev-requirements.txt`](./dev-requirements.txt).
  - The training/evaluation scripts are built on top of [MLFlow](https://mlflow.org/),
    and were developed with version 2.9.2

Installation
------------

The Argumentation Mining Transformer Module (AMTM) is installable as a Python
package. To do so, we recommend using some form of virtual environment first:

    $ python -m venv amtm-venv
    $ source ./amtm-venv/bin/activate
    (amtm-venv) $ pip install --upgrade pip setuptools wheel

Before installing the AMT Module package, we recommend you install your
preferred PyTorch version, for example, if you are running this from a machine
without GPU access, it's recommended to install PyTorch like:

    (amtm-venv) $ pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

If you don't explicitly install the version of PyTorch you'd like it's probably
that `pip` will try to install the default version for your system when
resolving the dependencies (e.g. for Linux, it's the GPU version).

After that, you can install the AMTM library like:

    (amtm-venv) $ pip install git+https://github.com/crscardellino/argumentation-mining-transformers@amtm-<VERSION>

Replacing `<VERSION>` with the version you want to install (>= 1.0.1-beta).
E.g.:

    (amtm-venv) $ pip install git+https://github.com/crscardellino/argumentation-mining-transformers@amtm-1.0.1-beta

**Note:** This installation of the AMTM Library will only install what's under
the `./amtm` directory so you can access it via `from amtm import *` under your
Python module development. This means that you can use it to directly access the
data modules, models and other utilities. If you want to use the training
scripts go to the [usage of the training and evaluation
tools](#usage-of-the-training-and-evaluation-tools) section.

### Pipeline

The `amtm.pipeline` module requires NLTK's PUNKT tokenizer. To install it run
the following command:

    (amtm-venv) $ python -m nltk.downloader punkt

Usage of the training and evaluation tools
------------------------------------------

### Local Installation

If you want to use the tools we offer for training and evaluation, you need
first to clone the repository in your local environment:

    $ git clone https://github.com/crscardellino/argumentation-mining-transformers/

After that, create the environment and install the package from the local copy
with the development requirements as well:

    $ python -m venv amtm-venv
    $ source ./amtm-venv/bin/activate
    (amtm-venv) $ pip install --upgrade pip setuptools wheel
    (amtm-venv) $ pip install -e ".[dev]"

### Scripts for Traning and Evaluation

There are 2 Python scripts, under the `./scripts` directory, ready to use:
[`./scripts/train.py`](./scripts/train.py) and
[`./scripts/eval.py`](./scripts/eval.py).

There are 4 bash scripts with examples of how to use each of these scripts:
[`./train_rel_class.sh`](./train_rel_class.sh), [`./train_seq_tag.sh`](./train_seq_tag.sh),
[`./eval_rel_class.sh`](./eval_rel_class.sh) and [`./eval_seq_tag.sh`](./eval_seq_tag.sh).

### Training

The Python train script runs the training and validation loops using Lightning.
It requires the following parameters:

    --train-data TRAIN_DATA
                        The train dataset path. It should already be in the
                        format for the corresponding task (`--task-type`).
    --output-dir OUTPUT_DIR
                        The directory where the model logs and checkpoints
                        will be stored.
    --task-type {rel-class,seq-tag}
                        Type of task. Use one of: rel-class, seq-tag
    --model MODEL       Either the name of one of the available models: bert,
                        deberta-v3, roberta, tiny-bert; or a Hugging Face
                        model. The HF model can be either a model available at
                        the HF Hub, or a model path. To load a checkpoint
                        reached using this same trainer script please use the
                        `--load-from-checkpoint` option.

The `train-data` file should be in the format corresponding to the task (tsv for
rel-class and conll for seq-tag). The `output-dir` is the directory where MLFlow
will store both the results and the model checkpoints. Finally, `model` is the
Hugging Face model to use (as a single string).

There are other options available as well:

    --validation-data VALIDATION_DATA
                        The validation dataset path. It should already be in
                        the format for the corresponding task (`--task-type`).
    --config CONFIG     Pretrained config name or path (if not the same as
                        `model`).
    --tokenizer TOKENIZER
                        Pretrained tokenizer name or path (if not the same as
                        `model`).
    --cache-dir CACHE_DIR
                        Directory for Hugging Face downloaded models.
    --load-from-checkpoint LOAD_FROM_CHECKPOINT
                        Path to a checkpoint file to continue training.
    --experiment-name EXPERIMENT_NAME
                        Suffix of MLFlow experiment.
    --run-name RUN_NAME Prefix of MLFlow run.
    --labels [LABELS ...]
                        The list of labels (separated by spaces) for the task.
                        If not given it will fallback to the default labels
                        for the task.
    --accelerator ACCELERATOR
                        What device to use as accelerator (cpu, gpu, tpu,
                        etc).
    --num-devices NUM_DEVICES
                        Number of devices to use. If not given selects
                        automatically.
    --num-workers NUM_WORKERS
                        Number of workers to use for DataLoaders. Set to -1 to
                        use all cpus.
    --epochs EPOCHS     Total number of training epochs to perform.
    --early-stopping EARLY_STOPPING
                        If > 0 then stops if there are `early-stopping` logs
                        without improvement on the validation loss.
    --batch-size BATCH_SIZE
                        Batch size (per GPU/CPU) for training.
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
    --lower-case        Should be active for lowercase transformers.
    --learning-rate LEARNING_RATE
                        The initial learning rate for AdamW.
    --weight-decay WEIGHT_DECAY
                        Weight decay for parameters that accept it.
    --warmup-steps WARMUP_STEPS
                        Number of steps for linear warmup.
    --weighted-loss     Only useful for Relationship Classification trainings.
                        If true the loss function is weighted inversely by
                        class.
    --log-every-n-steps LOG_EVERY_N_STEPS
                        Log every N update steps.
    --save-every-n-steps SAVE_EVERY_N_STEPS
                        Save checkpoint every N update steps.
    --random-seed RANDOM_SEED
                        Initial random seed.
    --fp16              Whether to use 16-bit (mixed) precision
    --debug             Set for debug mode.

If the `validation-data` file is not given, the training loop will not do
validation (nor early stopping). The `--debug` mode runs the experiment on a
tenth of the data, and is only useful to debug the script (e.g. to check
everything is working after installation).

Although not required, the parameters `experiment-name` and `run-name` are for
MLFlow and are recommended to be set to better differentiate between
models, especially if you are trying to run several evaluations over certain
models. The bash scripts with examples have a better indication of what to do.

The files `./train_rel_class.sh` and `./train_seq_tag.sh` show examples for
running train (and also evaluation) over the
[`./data/neoplasm`](./data/neoplasm/) dataset, both for Relation Classification
and Component Detection respectively.

### Evaluation

The Python evaluation script runs the evaluation over some test data, it will
look for the last trained model with the metadata given by `experiment-name` and
`run-name` and run the evaluation using one or all the checkpoints for that
model. It requires the following parameters:

    --test-data TEST_DATA
                        The evaluation dataset path. It should already be in
                        the format for the corresponding task (`--task-type`).
    --output-dir OUTPUT_DIR
                        The output directory where the model predictions will
                        be stored. In order to eval trained checkpoint, this
                        should match the output directory of the train script.
    --task-type {rel-class,seq-tag}
                        Type of task. Use one of: rel-class, seq-tag
    --model MODEL       Either the name of one of the available models: bert,
                        deberta-v3, roberta, tiny-bert; or a Hugging Face
                        model. The HF model can be either a model available at
                        the HF Hub, or a model path.

Where `test-data` has the same format as the train data for the task type, and
the `output-dir` points to the same directory used by mlflow during training.
The `model` should also be the same for correct evaluation.

Other optional parameters are the following:

    --config CONFIG     Pretrained config name or path (if not the same as
                        `model`).
    --tokenizer TOKENIZER
                        Pretrained tokenizer name or path (if not the same as
                        `model`).
    --cache-dir CACHE_DIR
                        Directory for Hugging Face downloaded models.
    --eval-without-checkpoint
                        If active, it will evaluate the model directly from HF
                        hub.
    --experiment-name EXPERIMENT_NAME
                        Suffix of MLFlow experiment.
    --run-name RUN_NAME   Prefix of MLFlow run.
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
    --num-workers NUM_WORKERS
                        Number of workers to use for DataLoaders. Set to -1 to
                        use all cpus.
    --batch-size BATCH_SIZE
                        Batch size for evaluation.
    --max-seq-length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        tokenization.Sequences longer than this will be
                        truncated, sequences shorter will be padded. If left
                        empty it will truncate to the model's max size and pad
                        to the maximum size of each training step.
    --lower-case        Should be active for lowercase transformers.
    --eval-all-checkpoints
                        Evaluate all checkpoints for the model.
    --random-seed RANDOM_SEED
                        Initial random seed.
    --weighted-loss     Only useful for Relationship Classification trainings.
                        If true the loss function is weighted inversely by
                        class.
    --fp16              Whether to use 16-bit (mixed) precision
    --debug             Set for debug mode.

The `--eval-without-checkpoint` is useful to evaluate a Hugging Face model
"out-of-the-box".  The `--relevant-labels` should be a subset of the `--labels`
and refers to the labels that require special treatment or evaluation, e.g. for
the case of Component Detection, relevant labels are the ones that start with
`B-` or `I-`.

Unlike the training script, even if we use a GPU accelerator, at the moment it's
only possible to use a single device for inference, since MultiGPU evaluation
it's difficult and sometimes
[ill-defined](https://github.com/Lightning-AI/pytorch-lightning/issues/8375).

The `--eval-all-checkpoints` flag is to run an evaluation for each of the checkpoints resulting from the training experiment run. If this flag is not set,
it only runs evaluation on the last checkpoint.

### MLFlow UI

After the evaluation is finished, you need to run the MLFlow UI server and access
the Web UI:

    (amtm-venv) $ mlflow ui --backend-uri $OUTPUT_DIR

Replace the `$OUTPUT_DIR` with the directory where the results were stored for
training and evaluation. Then access [the web UI](http://localhost:5000).

The experiments' names have the following structures:
`{TASK_TYPE}/{MODEL_NAME}/{train|eval}/{EXPERIMENT_NAME}`.
Depending on the `train` or `eval`, there are different recorded metrics:

- The `train` experiments only record the training loss and, optionally, the
  validation loss. They also have the model checkpoints logged as artifacts.
- The `eval` experiments record accuracy and F1-score (micro and macro) over all
  and only relevant labels. Besides, they log as artifacts the classification
  report (for all and only relevant), the confusion matrix (numeric and as a
  heatmap) and the predictions. For the case of Sequence Tagging, it also
  reports the [Seqeval Metrics](https://github.com/chakki-works/seqeval).

For the runs, the names have the following structure: `{RUN_NAME}/{TIMESTAMP}`.
The evaluation runs log among their parameters the `run_id` of the training they
are based on. They also provide a link in their description to that training
run.

#### Retrieving MLFlow Results from Remote Server

If you happen to run your experiments (both for train and evaluation) on a
remote server, and you want to run the MLFlow UI but don't have access to an
open port in the remote server, one solution is to export the results directory
to your local machine and run it locally. To do so, one solution is first to
`rsync` the output directory from the remote server to your machine:

    $ rsync -avzP user@remote.server:path/to/output/ ./output/

If you want to avoid moving the large checkpoint files that are stored in the
remote server, you can `rsync` like:

    $ rsync -avzP --exclude="*.ckpt" user@remote.server:path/to/output/ ./output/

However, before running the MLFlow UI on `./output/` you need to
[update the artifacts path](https://github.com/mlflow/mlflow/issues/3144) to
your local environment. The script
[`./scripts/update_artifacts_uri.py`](./scripts/update_artifacts_uri.py) can
do that for you:

    (amtm-venv) $ ./scripts/update_artifacts_uri.py --mlflow-uri ./output/

In this case, the `OUTPUT_DIR`` is `output/`, but it can be replaced with
whatever directory name you prefer.

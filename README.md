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
handle the latest [Hugging Face models](https://huggingface.co/models)
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
  - [PyTorch](https://pytorch.org) >= 2: Developed with 2.0.1
  - [Lightning](https://lightning.ai/) >= 2: Developed with 2.0.9
  - [Hugging Face](https://huggingface.co/) >= 4: Developed with 4.33.2
  - [Pytorch-CRF](https://pytorch-crf.readthedocs.io/en/stable/): Developed with
    0.7.2

Installation
------------

Starting from version 0.1.2, the acta module is installable as a Python package.
To do so, we recommend use some form of virtual environment first:

    $ python -m venv acta-venv
    $ source ./acta-venv/bin/activate
    (acta-venv) $ pip install --upgrade pip

Before installing the ACTA Module package, we recommend you to install your preferred
PyTorch version, for example, if you are running this from a machine without GPU access,
it's recommended to install PyTorch like:

    (acta-venv) $ pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

If you don't explicitly install the version of PyTorch you'd like it's probably
that `pip` will try to install the default version for your system when resolving
the dependencies (e.g. for Linux it's the GPU version).

After that, you can install the ACTA module:

    (acta-venv) $ pip install git+https://gitlab.com/wimmics-antidote/antidote-acta@acta-module-<VERSION>

Replacing `<VERSION>` with the version you want to install (>= 0.1.2). E.g.:

    (acta-venv) $ pip install git+https://gitlab.com/wimmics-antidote/antidote-acta@acta-module-0.1.2

### Pipeline

The `acta.pipeline` module requires NLTK's PUNKT tokenizer. In order to install it run the
following command:

    (acta-venv) $ python -m nltk.downloader punkt

Usage
-----

We provide a python script to train and evaluate models:
[`run_acta.py`](./run_acta.py), for a detailed list of options please run:

    python run_acta.py --help

We provide two example bash scripts for training and evaluation that use the
`run_acta.py` script:

- [`run_acta_single_device.sh`](./run_acta_single_device.sh) is for training and
  evaluation when there's only a single Non-CPU device (e.g. GPU). The script
  runs the `run_acta.py` script a single time and does training and evaluation
  in that same script.
- [`run_acta_multiple_devices.sh`](./run_acta_multiple_devices.sh) is for
  training and evaluation when there are multiple Non-CPU devices available for
  training. In this case it runs the `run_acta.py` script 2 times, first for
  training with multiple Non-CPU devices and then for evaluation but using a
  single device.
  Check [Multiple Devices Evaluation](#multiple-devices-evaluation) for more info.

### Multiple Devices Evaluation

When training a model using multiple Non-CPU devices (e.g. GPUs), the strategy
used by Pytorch Lightning is DistributedDataParallelism (or DDP), this means
that the same process will be run multiple times (instead of spawning childs
from a main process).

In this scenario, training and evaluating during the same run will have
unexpected behavior, since there can be race conditions when trying to write to
the same results files. As such, we advice to avoid training and evaluating
during the same process when having multiple Non-CPU devices. And in order to
avoid any problems, after the training is finished, you should run the
evaluation with a single device (check the `--num-devices` option in the
`run_acta.py` script).

For more information check
[this Github Issue](https://github.com/Lightning-AI/pytorch-lightning/issues/8375).
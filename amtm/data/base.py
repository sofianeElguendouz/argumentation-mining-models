"""
Module that defines the abstract class `BaseDataset` which is the base class for
`RelationClassificationDataset` and `SequenceTaggingDataset` in the module, and
the abstract class `BaseDataModule` which is the base class for
`RelationClassificationDataModule` and `SequenceTaggingDataModule`.

    Argumentation Mining Transformers Base Data Module
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

import logging

from abc import ABCMeta, abstractmethod
from pathlib import PosixPath
from lightning.pytorch import LightningDataModule
from multiprocessing import cpu_count
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    BaseDataset abstract class. Contains some common implementations for datasets classes.

    Parameters
    ==========
    tokenizer: AutoTokenizer
        Hugging Face Tokenizer.
    path_to_dataset: str
        Path to a dataset (the format depends on the type of dataset)
    label2id: Optional[Dict[str, int]]
        The mapping between labels and indices. If not provided it will be taken from
        the dataset, with a warning that not all labels might be available.
    id2label: Optional[Dict[int, str]]
        The reverse mapping of label2id that map the indices to the labels. If
        not given it will be taken by reversing label2id.
    max_seq_length: Optional[int]
        If > 0 truncates and pads each sequence of the dataset to `max_seq_length`.
    truncation_strategy: str
        What truncation strategy to use (by default truncates the longest
        sentence). Must be one of `longest_first`, `only_second`, `only_first`.
        Check https://huggingface.co/docs/transformers/pad_truncation for more
        information.
    **kwargs
        Extra keyword arguments dependant on the children classes. Used for the
        `_load_dataset` method among other class specific use cases.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        path_to_dataset: str,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
        max_seq_length: Optional[int] = None,
        truncation_strategy: str = "longest_first",
        **kwargs,
    ):
        super().__init__()
        assert truncation_strategy in {"longest_first", "only_second", "only_first"}

        self.tokenizer = tokenizer

        self.label2id = label2id
        if self.label2id is None:
            logger.warning(
                "There label2id parameter is missing, it will be calculated from the dataset. "
                "This might not have all the available labels."
            )
        self._load_dataset(path_to_dataset, **kwargs)

        self.id2label = id2label if id2label else {idx: lbl for lbl, idx in self.label2id.items()}
        self.max_seq_length = max_seq_length
        self.truncation_strategy = truncation_strategy

    @abstractmethod
    def _load_dataset(self, path_to_dataset: str, **kwargs):
        """
        Method to load the dataset to `self.dataset` as well as other attributes.
        Must be implemented on each class that inherits from this one.

        Parameters
        ==========
        path_to_dataset: str
            Path to the dataset to load (it comes from the class constructor).
        **kwargs
            Extra arguments that are class specific for children classes.
        """


class BaseDataModule(LightningDataModule, metaclass=ABCMeta):
    """
    BaseDataModule abstract class. Contains some common implementations
    for data module classes.

    Check the documentation for LightningDataModule for more information:
    https://lightning.ai/docs/pytorch/stable/data/datamodule.html

    Parameters
    ==========
    data_splits: Dict[str, PosixPath]
        Mapping between splits and paths to the files corresponding to such
        splits. It must have at least 1 split otherwise it will raise
        ValueError. The data split must be one of {'train', 'test',
        'validation'}.
    tokenizer_name_or_path: str
        Name or path to a Hugging Face Tokenizer to load.
    labels: List[str]
        The set of labels (that will be returned by the property `labels`).  The
        labels should be unique and sorted (each label will correspond to the
        index it belongs in the list, starting by 0). If not given, it will
        resort to the class attribute `LABELS` which should be defined for each
        subclass.
    tokenizer_config: Dict[str, Any]
        Extra tokenizer specific config (e.g. do_lower_case, cache_dir, use_fast, etc)
    datasets_config: Dict[str, Any]
        A set of configurations specific for the datasets.
    train_batch_size: int
        Size of the training batches (per GPU/CPU if distributed).
    eval_batch_size: int
        Size of the evaluation batches (per GPU/CPU if distributed).
    evaluation_split: Optional[str]
        The split to use for evaluation. If given, it must be one of {'train',
        'test', 'validation'}.
    num_workers: int
        Number of workers to use. If < 0 uses all CPUs.
    """

    def __init__(
        self,
        data_splits: Dict[str, PosixPath],
        tokenizer_name_or_path: str,
        labels: Optional[List[str]],
        tokenizer_config: Dict[str, Any] = dict(use_fast=True),
        datasets_config: Dict[str, Any] = dict(max_seq_length=128),
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        evaluation_split: Optional[str] = None,
        num_workers: int = -1,
    ):
        super().__init__()

        if len(data_splits) == 0:
            raise ValueError("The `data_splits` argument must not be empty.")

        valid_splits = {"train", "test", "validation"}
        if not valid_splits.issuperset(data_splits.keys()):
            raise ValueError(f"The data splits must be one of: {', '.join(valid_splits)}")
        if evaluation_split is not None and evaluation_split not in valid_splits:
            raise ValueError(f"The evaluation split must be in one of: {', '.join(valid_splits)}")

        if labels is not None and len(labels) != len(set(labels)):
            raise ValueError("The labels should be unique.")

        self.data_splits = data_splits
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_config)
        self.datasets = {}
        self.datasets_config = datasets_config
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer_config = tokenizer_config
        self.evaluation_split = evaluation_split
        self.num_workers = num_workers if num_workers > 0 else cpu_count()

        if labels is None:
            labels = self.LABELS
        self._labels = {lbl: idx for idx, lbl in enumerate(labels)}

        # Silencing the warning to pad with fast tokenizer since with Lightning this is
        # not possible unless we pad beforehand
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    @classmethod
    @property
    @abstractmethod
    def LABELS(cls) -> List[str]:
        """
        This should be overridden by the subclasses (as a class attribute),
        otherwise it will fail when trying to instantiate in the `__init__`
        method if no `labels` attribute is given.
        """

    @property
    def labels(self) -> Dict[str, int]:
        """
        Property that displays the map between labels and ids for the Datasets.

        Returns
        =======
        Dict[str, int]
            Mapping between a label and its corresponding numerical id.
        """
        return self._labels

    @property
    def label2id(self) -> Dict[str, int]:
        """
        Proxy method to access one of the datasets `label2id` which can differ
        from the DataModule `labels` property (e.g. by having extra labels such
        as the extension label `X` or the padding label `PAD`).

        It will try to access based on a priority. First the train dataset, if
        it's not present, it will use the evaluation_split dataset. An finally
        it will check for any dataset present.

        If there's no dataset it raises an error.
        """
        if len(self.datasets) == 0:
            raise ValueError("The datasets are not present. Please run `setup`.")

        if "train" in self.datasets:
            # First try with the training dataset
            return self.datasets["train"].label2id
        elif self.evaluation_split in self.datasets:
            # Check the evaluation dataset
            return self.datasets[self.evaluation_split].label2id
        else:
            # Return whatever dataset that is present as a last resource
            return list(self.datasets.values())[0].label2id

    @property
    def id2label(self) -> Dict[str, int]:
        """
        Proxy method to access one of the datasets `id2label`.

        It will try to access based on a priority. First the train dataset, if
        it's not present, it will use the evaluation_split dataset. An finally
        it will check for any dataset present.

        If there's no dataset it raises an error.
        """
        if len(self.datasets) == 0:
            raise ValueError("The datasets are not present. Please run `setup`.")

        if "train" in self.datasets:
            # First try with the training dataset
            return self.datasets["train"].id2label
        elif self.evaluation_split in self.datasets:
            # Check the evaluation dataset
            return self.datasets[self.evaluation_split].id2label
        else:
            # Return whatever dataset that is present as a last resource
            return list(self.datasets.values())[0].id2label

    @property
    @abstractmethod
    def collate_fn(self) -> Callable:
        """
        Returns the collate function. It depends on the type of dataset.

        Returns
        =======
        Callable
            A function or class with the __call__ method implemented.
        """

    def prepare_data(self):
        """
        Method to prepare data. It is called only once across all devices.
        """
        AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, **self.tokenizer_config)

    def setup(self, stage: str):
        """
        Method to load all the dataset splits to self.datasets. It is called
        for every device.

        Parameters
        ==========
        stage: str
            One of `fit`, `test`, `validate` and `predict`. Only for
            compatibility.
        """
        for split, path in self.data_splits.items():
            self.datasets[split] = self._load_dataset_split(path)

    @abstractmethod
    def _load_dataset_split(self, path_to_dataset: str) -> BaseDataset:
        """
        Method to load a dataset split as a part of self.datasets.
        Must be implemented on each class that inherits from this one.

        Parameters
        ==========
        path_to_dataset: str
            Path to the dataset to load.
        """

    def train_dataloader(self) -> Optional[DataLoader]:
        """
        Returns the dataloader for train (if it exists) otherwise returns None.
        """
        if "train" in self.datasets:
            return DataLoader(
                dataset=self.datasets["train"],
                batch_size=self.train_batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
            )
        else:
            None

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Returns the dataloader for validation (if it exists) otherwise returns None.
        """
        if "validation" in self.datasets:
            return DataLoader(
                dataset=self.datasets["validation"],
                batch_size=self.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
            )
        else:
            return None

    def test_dataloader(self) -> Optional[DataLoader]:
        """
        Returns a DataLoader for the selected `evaluation_split`. If there's no
        evaluation split, returns None.
        """
        if self.evaluation_split:
            return DataLoader(
                dataset=self.datasets[self.evaluation_split],
                batch_size=self.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
            )
        else:
            return None

    def predict_dataloader(self) -> Optional[DataLoader]:
        """
        Returns a DataLoader for the selected `evaluation_split` (for
        prediction). If there's no evaluation split, returns None.
        """
        if self.evaluation_split:
            return DataLoader(
                dataset=self.datasets[self.evaluation_split],
                batch_size=self.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
            )
        else:
            return None

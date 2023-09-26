"""
Module that defines the abstract class `BaseDataset` which is the base class for
`RelationClassificationDataset` and `SequenceTaggingDataset` in the module.

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

import logging

from abc import ABCMeta, abstractmethod
from pathlib import PosixPath
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    BaseDataset abstract class. Contains some common implementations for datasets classes.

    Parameters
    ----------
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
    max_seq_lenght: Optional[int]
        If > 0 truncates and pads each sequence of the dataset to `max_seq_length`.
    truncation_strategy: str
        What truncation strategy to use (by default truncates the longest
        sentence). Must be one of `longest_first`, `only_second`, `only_first`.
    """
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 path_to_dataset: str,
                 label2id: Optional[Dict[str, int]] = None,
                 id2label: Optional[Dict[int, str]] = None,
                 max_seq_length: Optional[int] = None,
                 truncation_strategy: str = 'longest_first',
                 **kwargs):
        super().__init__()
        assert truncation_strategy in {'longest_first', 'only_second', 'only_first'}

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
        ----------
        path_to_dataset: str
            Path to the dataset to load (it comes from the class constructor).
        """


class BaseDataModule(LightningDataModule):
    """
    BaseDataModule abstract class. Contains some common implementations
    for data module classes.

    Parameters
    ----------
    data_splits: Dict[str, PosixPath]
        Mapping between splits and paths to the files corresponding to such
        splits.
    tokenizer_name_or_path: str
        Name or path to a Hugging Face Tokenizer to load.
    tokenizer_config: Dict[str, Any]
        Extra tokenizer specific config (e.g. do_lower_case, cache_dir, use_fast, etc)
    datasets_config: Dict[str, Any]
        A set of configurations specific for the datasets.
    train_batch_size: int
        Size of the training batches (per GPU/CPU if distributed).
    eval_batch_size: int
        Size of the evaluation batches (per GPU/CPU if distributed).
    evaluation_split: Optional[str]
        The split to use for evaluation.
    """
    def __init__(self,
                 data_splits: Dict[str, PosixPath],
                 tokenizer_name_or_path: str,
                 tokenizer_config: Dict[str, Any] = dict(use_fast=True),
                 datasets_config: Dict[str, Any] = dict(max_seq_lenght=128),
                 train_batch_size: int = 8,
                 eval_batch_size: int = 8,
                 cache_dir: Optional[str] = None,
                 evaluation_split: Optional[str] = None):
        super().__init__()
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_config)
        self.datasets = {}
        self.datasets_config = datasets_config
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer_config = tokenizer_config
        self.evaluation_split = evaluation_split

    @property
    @abstractmethod
    def labels(self) -> Dict[str, int]:
        """
        Property that defines the map betwenn labels and ids for the Datasets.

        Returns
        -------
        Dict[str, int]
            Mapping between a label and its corresponding numerical id.
        """

    @property
    def label2id(self) -> Dict[str, int]:
        """
        Proxy method to access one of the datasets `label2id` which can differ
        from the DataModule labels (e.g. by having extra labels such as the
        extension label `X` or the padding label `PAD`).

        If there's no dataset it raises an error.
        """
        if len(self.datasets) == 0:
            raise ValueError("The datasets are not set")

        if 'train' in self.datasets:
            # First try with the training dataset
            return self.datasets['train'].label2id
        elif self.evaluation_split in self.datasets:
            # Check the evaluation dataset
            return self.datasets[self.evaluation_split].label2id
        else:
            # Return whatever dataset that is present as a last resource
            return list(self.datasets.values())[0].label2id

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
        ----------
        stage: str
            One of `fit`, `test`, `validate` and `predict`
        """
        for split, path in self.data_splits.items():
            self.datasets[split] = self._load_dataset_split(path)

    @abstractmethod
    def _load_dataset_split(self, path_to_dataset: str) -> BaseDataset:
        """
        Method to load a dataset split as a part of self.datasets.
        Must be implemented on each class that inherits from this one.

        Parameters
        ----------
        path_to_dataset: str
            Path to the dataset to load.
        """

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader object for the train split of the dataset.
        """
        return DataLoader(self.datasets["train"],
                          batch_size=self.train_batch_size,
                          shuffle=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Returns the dataloader for validation (if it exists) otherwise returns None.
        """
        if "validation" in self.datasets:
            return DataLoader(self.datasets["validation"],
                              batch_size=self.eval_batch_size,
                              shuffle=False)
        else:
            return None

    def test_dataloader(self) -> Optional[DataLoader]:
        """
        Returns a DataLoader for the selected `evaluation_split`. If there's no
        evaluation split, returns None.
        """
        if self.evaluation_split:
            return DataLoader(self.datasets[self.evaluation_split],
                              batch_size=self.eval_batch_size,
                              shuffle=False)
        else:
            return None

    def predict_dataloader(self) -> Optional[DataLoader]:
        """
        Returns a DataLoader for the selected `evaluation_split` (for
        prediction). If there's no evaluation split, returns None.
        """
        if self.evaluation_split:
            return DataLoader(self.datasets[self.evaluation_split],
                              batch_size=self.eval_batch_size,
                              shuffle=False)
        else:
            return None

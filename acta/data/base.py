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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, Optional


logger = logging.getLogger(__name__)


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    BaseDataset abstract class. Contains some common implementations for datasets classes.

    Parameters
    ----------
    tokenizer_model_or_path: str
        Name or path to a Hugging Face Tokenizer to load.
    path_to_dataset: str
        Path to a dataset (the format depends on the type of dataset)
    label2id: Optional[Dict[str, int]]
        The mapping between labels and indices. If not provided it will be taken from
        the dataset, with a warning that not all labels might be available.
    id2label: Optional[Dict[int, str]]
        The reverse mapping of label2id that map the indices to the labels. If
        not given it will be taken by reversing label2id.
    """
    def __init__(self,
                 tokenizer_model_or_path: str,
                 path_to_dataset: str,
                 label2id: Optional[Dict[str, int]] = None,
                 id2label: Optional[Dict[int, str]] = None,
                 **kwargs):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_or_path)
        self.label2id = label2id

        if self.label2id is None:
            logger.warning(
                "There labels parameter is missing, it will be calculated from the dataset. "
                "This might not have all the available labels."
            )
        self._load_dataset(path_to_dataset, **kwargs)
        self.id2label = id2label if id2label else {idx: lbl for lbl, idx in self.label2id.items()}

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
    model_model_or_path: str
        Name or path to a Hugging Face Model to load.
    tokenizer_model_or_path: str
        Name or path to a Hugging Face Tokenizer to load.
    path_to_dataset: str
        Path to a dataset (the format depends on the type of dataset)
    """
    def __init__(self,
                data_splits: Dict[str, PosixPath],
                model_name_or_path: str,
                tokenizer_name_or_path: str,
                max_seq_length: int = 128,
                train_batch_size: int = 32,
                eval_batch_size: int = 32
                ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.datasets = {}
        self.num_labels = None
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    @abstractmethod
    def _load_dataset_split(self,
                            path_to_dataset: str,
                            **kwargs):
        """
        Method to load a dataset split as a part of self.dataset.
        Must be implemented on each class that inherits from this one.

        Parameters
        ----------
        path_to_dataset: str
            Path to the dataset to load (it comes from the class constructor).
        """
        pass

    def _load_dataset_splits(self):
        """
        Method to load all the dataset splits to self.dataset.

        Parameters
        ----------
        None
        """
        for split, path in self.data_splits.items():
            self.datasets[split] = self._load_dataset_split(path)

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        """
        Returns a DataLoader object for the train split of the dataset.

        Parameters
        ----------
        None
        """
        return DataLoader(self.datasets["train"], batch_size=self.train_batch_size, shuffle=True)

    def dev_dataloader(self):
        """
        Returns one or more DataLoader object/s for the dev split of the
        dataset.

        Parameters
        ----------
        None
        """
        if len(self.eval_splits) == 1:
            return DataLoader(self.datasets["eval"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.datasets[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        """
        Returns one or more DataLoader object/s for the test split of the
        dataset.

        Parameters
        ----------
        None
        """
        if len(self.eval_splits) == 1:
            return DataLoader(self.datasets["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.datasets[x], batch_size=self.eval_batch_size) for x in self.eval_splits]
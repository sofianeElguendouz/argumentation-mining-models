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
from torch.utils.data import Dataset
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
    labels: Optional[Dict[str, int]]
        The mapping between labels and indices. If not provided it will be taken from
        the dataset, with a warning that not all labels might be available.
    """
    def __init__(self,
                 tokenizer_model_or_path: str,
                 path_to_dataset: str,
                 labels: Optional[Dict[str, int]] = None,
                 **kwargs):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_or_path)
        self.labels = labels

        if self.labels is None:
            logger.warning(
                "There labels parameter is missing, it will be calculated from the dataset. "
                "This might not have all the available labels."
            )
        self._load_dataset(path_to_dataset, missing_labels=self.labels is None, **kwargs)

    @abstractmethod
    def _load_dataset(self, path_to_dataset: str, missing_labels: bool = False, **kwargs):
        """
        Method to load the dataset to `self.dataset` as well as other attributes.
        Must be implemented on each class that inherits from this one.

        Parameters
        ----------
        path_to_dataset: str
            Path to the dataset to load (it comes from the class constructor).
        missing_labels: bool
            If True, then collect the labels from the dataset file as well.
        """

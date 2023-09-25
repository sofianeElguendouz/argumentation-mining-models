"""
RelationClassificationDataset module. It has the definition of the dataset to read
column based data (csv, tsv) for Relation classification.

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

import csv

from transformers.tokenization_utils_base import BatchEncoding
from typing import Dict, Optional

from .base import BaseDataset, BaseDataModule


class RelationClassificationDataset(BaseDataset):
    """
    Dataset for classification of relationship between a pairs of sentences
    (e.g. supports, attacks, etc.).

    It expects the dataset to be in column format (csv, tsv, etc.) with the
    first column being the label (it might have some extra info such as the
    __label__ prefix).

    Parameters
    ----------
    tokenizer_model_or_path: str
        Refer to BaseDataset.
    path_to_dataset: str
        Refer to BaseDataset.
    label2id: Optional[Dict[str, int]]
        Refer to BaseDataset.
    id2label: Optional[Dict[int, str]]
        Refer to BaseDataset.
    delimiter: str
        Character used to split the columns in the dataset (comma, colon, tab,
        etc).
    quotechar: str
        Character that is used when a delimiter is present in a column but it
        shouldn't split the column.
    label_prefix: str
        A prefix to be removed from the label2id in the dataset.
    """
    def __init__(self,
                 tokenizer_model_or_path: str,
                 path_to_dataset: str,
                 label2id: Optional[Dict[str, int]] = None,
                 id2label: Optional[Dict[int, str]] = None,
                 delimiter: str = '\t',
                 quotechar: str = '"',
                 label_prefix: str = '__label__'):
        super().__init__(tokenizer_model_or_path=tokenizer_model_or_path,
                         path_to_dataset=path_to_dataset,
                         label2id=label2id, id2label=id2label,
                         delimiter=delimiter, quotechar=quotechar,
                         label_prefix=label_prefix)

    def _load_dataset(self,
                      path_to_dataset: str,
                      delimiter: str = '\t',
                      quotechar: str = '"',
                      label_prefix: str = '__label__'):
        with open(path_to_dataset, 'rt') as fh:
            csv_reader = csv.reader(fh, delimiter=delimiter, quotechar=quotechar)
            dataset = list(csv_reader)

        self.dataset = [
            {
                "text": d[1],
                "text_pair": d[2]
            }
            for d in dataset
        ]

        target = [d[0].lstrip(label_prefix) for d in dataset]
        if self.label2id is None:
            self.label2id = {lbl: idx for idx, lbl in enumerate(sorted(set(target)))}

        self.target = [self.label2id[tgt] for tgt in target]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> BatchEncoding:
        data = self.dataset[idx]

        if isinstance(idx, slice):
            text = [d['text'] for d in data]
            text_pair = [d['text_pair'] for d in data]
            tokenized_data = self.tokenizer(text=text, text_pair=text_pair,
                                            padding=True, truncation=True)
        else:
            tokenized_data = self.tokenizer(**data, truncation=True)

        tokenized_data['label'] = self.target[idx]

        return tokenized_data

class RelationClassificationDataModule(BaseDataModule):
    """
    Data module for classification of relationship between a pairs of sentences
    (e.g. supports, attacks, etc.).

    Parameters
    ----------
    model_model_or_path: str
        Name or path to a Hugging Face Model to load.
    tokenizer_model_or_path: str
        Name or path to a Hugging Face Tokenizer to load.
    path_to_dataset: str
        Path to a dataset (the format depends on the type of dataset)
    """
    def _load_dataset_split(self,
                            path: str,
                            **kwargs):
        """
        Method to load a dataset split as a part of self.dataset.
        Must be implemented on each class that inherits from this one.

        Parameters
        ----------
        path: str
            Path to the dataset split to load (it comes from the class 
            constructor).
        """
        dataset = RelationClassificationDataset(tokenizer_model_or_path = self.tokenizer_name_or_path,
                                                        path_to_dataset = path # Adjust path when we have a final project structure
                                                )
        return dataset

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

from transformers import AutoTokenizer
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
    tokenizer: AutoTokenizer
        Refer to BaseDataset.
    path_to_dataset: str
        Refer to BaseDataset.
    label2id: Optional[Dict[str, int]]
        Refer to BaseDataset.
    id2label: Optional[Dict[int, str]]
        Refer to BaseDataset.
    max_seq_length: Optional[int]
        Refer to BaseDataset.
    truncation_strategy: str
        Refer to BaseDataset
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
                 tokenizer: AutoTokenizer,
                 path_to_dataset: str,
                 label2id: Optional[Dict[str, int]] = None,
                 id2label: Optional[Dict[int, str]] = None,
                 max_seq_length: Optional[int] = None,
                 truncation_strategy: str = 'longest_first',
                 delimiter: str = '\t',
                 quotechar: str = '"',
                 label_prefix: str = '__label__'):
        super().__init__(tokenizer=tokenizer,
                         path_to_dataset=path_to_dataset,
                         label2id=label2id, id2label=id2label,
                         max_seq_length=max_seq_length, truncation_strategy=truncation_strategy,
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
            tokenized_data = self.tokenizer(
                text=text, text_pair=text_pair,
                padding='max_length' if self.max_seq_length else True,
                truncation=self.truncation_strategy,
                max_length=self.max_seq_length
            )
        else:
            tokenized_data = self.tokenizer(
                **data,
                padding='max_length' if self.max_seq_length else False,
                truncation=self.truncation_strategy,
                max_length=self.max_seq_length
            )

        tokenized_data['label'] = self.target[idx]

        return tokenized_data


class RelationClassificationDataModule(BaseDataModule):
    """
    Data module for classification of relationship between a pairs of sentences
    (e.g. supports, attacks, etc.).
    """
    @property
    def labels(self) -> Dict[str, int]:
        return {
            "noRel": 0,
            "Support": 1,
            "Attack": 2
        }

    def _load_dataset_split(self, path_to_dataset: str) -> RelationClassificationDataset:
        return RelationClassificationDataset(
            tokenizer=self.tokenizer,
            path_to_dataset=path_to_dataset,
            label2id=self.labels,
            **self.datasets_config
        )

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

from .base import BaseDataset


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
    labels: Optional[Dict[str, int]]
        Refer to BaseDataset.
    delimiter: str
        Character used to split the columns in the dataset (comma, colon, tab,
        etc).
    quotechar: str
        Character that is used when a delimiter is present in a column but it
        shouldn't split the column.
    label_prefix: str
        A prefix to be removed from the labels in the dataset.
    """
    def __init__(self,
                 tokenizer_model_or_path: str,
                 path_to_dataset: str,
                 labels: Optional[Dict[str, int]] = None,
                 delimiter: str = '\t',
                 quotechar: str = '"',
                 label_prefix: str = '__label__'):
        super().__init__(tokenizer_model_or_path=tokenizer_model_or_path,
                         path_to_dataset=path_to_dataset, labels=labels,
                         delimiter=delimiter, quotechar=quotechar,
                         label_prefix=label_prefix)

    def _load_dataset(self,
                      path_to_dataset: str,
                      missing_labels: bool = False,
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
        if missing_labels:
            self.labels = {lbl: idx for idx, lbl in enumerate(sorted(set(target)))}

        self.target = [self.labels[tgt] for tgt in target]

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

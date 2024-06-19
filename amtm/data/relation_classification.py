"""
Relation Classification Datasets. It has the definitions of the
`RelationClassificationDataset` and the `RelationClassificationDataModule`.
The Dataset reads column based data (csv, tsv) for Relation Classification.

    Argumentation Mining Transformers Relation Classification Data Module
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

import csv

from collections import Counter
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers.tokenization_utils_base import BatchEncoding
from typing import Callable, Dict, List, Optional, Tuple, Union

from .base import BaseDataset, BaseDataModule


class RelationClassificationDataset(BaseDataset):
    """
    Dataset for classification of relationship between a pairs of sentences
    (e.g. supports, attacks, etc.).

    It expects the dataset to be in column format (csv, tsv, etc.) with the
    first column being the label (it might have some extra info such as the
    __label__ prefix).

    Parameters
    ==========
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

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        path_to_dataset: str,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
        max_seq_length: Optional[int] = None,
        truncation_strategy: str = "longest_first",
        delimiter: str = "\t",
        quotechar: str = '"',
        label_prefix: str = "__label__",
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            path_to_dataset=path_to_dataset,
            label2id=label2id,
            id2label=id2label,
            max_seq_length=max_seq_length,
            truncation_strategy=truncation_strategy,
            delimiter=delimiter,
            quotechar=quotechar,
            label_prefix=label_prefix,
            **kwargs,
        )

    def _load_dataset(
        self,
        path_to_dataset: str,
        delimiter: str = "\t",
        quotechar: str = '"',
        label_prefix: str = "__label__",
        **kwargs,
    ):
        with open(path_to_dataset, "rt") as fh:
            if delimiter == "\t":
                # Don't quote for the case of TSV files
                csv_reader = csv.reader(fh, delimiter=delimiter, quoting=csv.QUOTE_NONE)
            else:
                csv_reader = csv.reader(fh, delimiter=delimiter, quotechar=quotechar)
            dataset = list(csv_reader)

        self.dataset = [{"text": d[1], "text_pair": d[2]} for d in dataset]

        target = [d[0].lstrip(label_prefix) for d in dataset]
        if self.label2id is None:
            self.label2id = {lbl: idx for idx, lbl in enumerate(sorted(set(target)))}

        self.target = [self.label2id[tgt] for tgt in target]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> BatchEncoding:
        data = self.dataset[idx]

        if isinstance(idx, slice):
            text = [d["text"] for d in data]
            text_pair = [d["text_pair"] for d in data]
            tokenized_data = self.tokenizer(
                text=text,
                text_pair=text_pair,
                padding="max_length" if self.max_seq_length else True,
                truncation=self.truncation_strategy,
                max_length=self.max_seq_length,
            )
        else:
            tokenized_data = self.tokenizer(
                **data,
                padding="max_length" if self.max_seq_length else False,
                truncation=self.truncation_strategy,
                max_length=self.max_seq_length,
            )

        tokenized_data["label"] = self.target[idx]

        return tokenized_data


class RelationClassificationDataModule(BaseDataModule):
    """
    Data module for classification of relationship between a pairs of sentences
    (e.g. supports, attacks, etc.).
    """

    LABELS = ["noRel", "Support", "Attack"]

    @property
    def classes_weights(self) -> List[float]:
        """
        Returns the weights for the classes. The class weight is calculated as
        the inverse of the ratio of occurrences of each class. This way the less
        occurrences a class has, the more important it is.

        It will try to access based on a priority. First the train dataset, if
        it's not present, it will use the evaluation_split dataset. An finally
        it will check for any dataset present.

        If there's no dataset it raises an error.
        """
        if len(self.datasets) == 0:
            raise ValueError("The datasets are not present. Please run `setup`.")

        if "train" in self.datasets:
            # First try with the training dataset
            target = self.datasets["train"].target
        elif self.evaluation_split in self.datasets:
            # Check the evaluation dataset
            target = self.datasets[self.evaluation_split].target
        else:
            # Return whatever dataset that is present as a last resource
            target = list(self.datasets.values())[0].target

        class_counter = Counter(target)
        return [1 - class_counter[c] / sum(class_counter.values()) for c in sorted(class_counter)]

    @property
    def collate_fn(self) -> Callable:
        return DataCollatorWithPadding(self.tokenizer)

    def decode_predictions(
        self,
        input_ids: Union[List[int], List[List[int]]],
        predictions: Union[int, List[int]],
        labels: Union[int, List[int], None] = None,
    ) -> Union[Tuple[str], List[Tuple[str]]]:
        """
        Decodes the input_ids, which can be a single instance (List[int]) or a
        batch of instances (List[List[int]]) into its corresponding pair of
        sentences. It appends the predicted labels (using id2label) and, if
        present, the true labels.

        Parameters
        ==========
        input_ids: List[int] | List[List[int]]
            The tokens ids of a single instance or a batch of instances.
        predictions: int | List[Int]
            The predicted label for a single instance or the list of predicted
            labels for a batch of instances.
        labels: int | List[int], optional
            If given, the true labels.

        Return
        ======
        Tuple[str], List[Tuple[str]]
            Depending on the type of input, it can return a single tuple
            or a list of tuples (if a batch of instances). The tuple
            has the form: (predicted_label, sentence1, sentence2) or
            (true_label, predicted_label, sentence1, sentence2) if the
            true label is given.
        """
        if isinstance(input_ids[0], int):
            # Single input
            sentence1 = self.tokenizer.decode(
                input_ids[: input_ids.index(self.tokenizer.sep_token_id)], skip_special_tokens=True
            )
            sentence2 = self.tokenizer.decode(
                input_ids[input_ids.index(self.tokenizer.sep_token_id) :], skip_special_tokens=True
            )
            predicted_label = self.id2label[predictions]
            if labels is not None:
                true_label = self.id2label[labels]
                return true_label, predicted_label, sentence1, sentence2
            else:
                return predicted_label, sentence1, sentence2
        else:
            # Batch of inputs
            if labels is None:
                return [
                    (
                        self.id2label[prediction],
                        self.tokenizer.decode(
                            input_id[: input_id.index(self.tokenizer.sep_token_id)],
                            skip_special_tokens=True,
                        ),
                        self.tokenizer.decode(
                            input_id[input_id.index(self.tokenizer.sep_token_id) :],
                            skip_special_tokens=True,
                        ),
                    )
                    for prediction, input_id in zip(predictions, input_ids)
                ]
            else:
                return [
                    (
                        self.id2label[label],
                        self.id2label[prediction],
                        self.tokenizer.decode(
                            input_id[: input_id.index(self.tokenizer.sep_token_id)],
                            skip_special_tokens=True,
                        ),
                        self.tokenizer.decode(
                            input_id[input_id.index(self.tokenizer.sep_token_id) :],
                            skip_special_tokens=True,
                        ),
                    )
                    for label, prediction, input_id in zip(labels, predictions, input_ids)
                ]

    def _load_dataset_split(self, path_to_dataset: str) -> RelationClassificationDataset:
        """
        Check BaseDataModule for documentation.
        """
        return RelationClassificationDataset(
            tokenizer=self.tokenizer,
            path_to_dataset=path_to_dataset,
            label2id=self.labels,
            **self.datasets_config,
        )

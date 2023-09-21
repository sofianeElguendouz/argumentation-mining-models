import csv

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

        # Once we are sure the labels were loaded, we map the target to its
        # corresponding labels.
        self.target = [self.labels[t] for t in self.target]

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
        self.target = [d[0].lstrip(label_prefix) for d in dataset]

    def _get_labels(self) -> Dict[str, int]:
        return {lbl: idx for idx, lbl in enumerate(sorted(set(self.target)))}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
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

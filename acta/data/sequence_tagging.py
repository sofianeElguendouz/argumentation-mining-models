"""
Sequence Tagging Datasets. It has the definitions of the
`SequenceTaggingDataset` and the `SequenceTaggingDataModule`.
The Dataset reads CONLL based column format for Sequence Tagging (Token
Classification).

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

from itertools import chain
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers.tokenization_utils_base import BatchEncoding
from typing import Callable, Dict, List, Optional, Tuple, Union

from .base import BaseDataset, BaseDataModule

logger = logging.getLogger()


class SequenceTaggingDataset(BaseDataset):
    """
    Dataset for sequence tagging (i.e. classify each token in a sequence of
    tokens).

    It expects data in CONLL format, where each token is associated to a "BIO"
    style label.

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
    token_position: int
        The position (column, with start on 0) where the token is present in the
        CONLL dataset.
    label_position: int
        The position (column, with start on 0) where the label is present in the
        CONLL dataset.
    use_extension_label: bool
        If true, uses a special label 'X' for special tokens in the transformer
        (e.g. [CLS], [SEP], <s>, etc.) as well as subtokens (if
        copy_label_to_subtoken is false).  If false, it will use the 'PAD' label
        instead.
    copy_label_to_subtoken: bool
        If true, when a token is separated in subtokens by the tokenizer, it
        copies the label to each subtoken. If it false, the assigned label to
        each subtoken after the first will depend on the value of
        `use_extension_label`. If the extension label 'X' exists, it will use
        it, otherwise it will use the special 'PAD' label.
    """
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 path_to_dataset: str,
                 label2id: Optional[Dict[str, int]] = None,
                 id2label: Optional[Dict[int, str]] = None,
                 max_seq_length: Optional[int] = None,
                 truncation_strategy: str = 'longest_first',
                 delimiter: str = '\t',
                 token_position: int = 1,
                 label_position: int = 4,
                 use_extension_label: bool = False,
                 copy_label_to_subtoken: bool = True,
                 **kwargs):
        super().__init__(tokenizer=tokenizer,
                         path_to_dataset=path_to_dataset,
                         label2id=label2id, id2label=id2label,
                         max_seq_length=max_seq_length, truncation_strategy=truncation_strategy,
                         delimiter=delimiter, token_position=token_position,
                         label_position=label_position, **kwargs)

        self.use_extension_label = use_extension_label
        self.copy_label_to_subtoken = copy_label_to_subtoken

        if 'PAD' not in self.label2id and 0 in self.id2label:
            replace_label = self.id2label[0]
            logger.warning(f"Replacing the id of label '{replace_label}' because 0 is reserved "
                           f"for special padding label 'PAD'. The new id of '{replace_label}' "
                           f"will be '{len(self.label2id)}'")
            self.label2id[replace_label] = len(self.label2id)
            self.id2label[self.label2id[replace_label]] = replace_label
            self.label2id['PAD'] = 0
            self.id2label[0] = 'PAD'
        elif 'PAD' not in self.label2id:
            logger.warning("Prepending the special label 'PAD'.")
            self.label2id['PAD'] = 0
            self.id2label[0] = 'PAD'

        if self.use_extension_label and 'X' not in self.label2id:
            # Add the extension label
            self.label2id['X'] = len(self.label2id)
            self.id2label[self.label2id['X']] = 'X'

    def _load_dataset(self,
                      path_to_dataset: str,
                      missing_labels: bool = False,
                      delimiter: str = '\t',
                      token_position: str = 1,
                      label_position: str = 4,
                      **kwargs):
        """
        Loads a dataset in CONLL format. The CONLL file is expected to have the
        traditional format of one word/token (with its corresponding columns,
        among of which is the label) per each line. Single blank lines serve to
        separate sentences and double blank lines serve to separate paragraphs.
        """
        with open(path_to_dataset, 'rt') as fh:
            sentences = []
            sentence_tokens = []
            sentence_labels = []
            for line in fh:
                line = line.strip().split(delimiter)

                if len(line) < 2:
                    # We have the end of a sentence
                    assert len(sentence_tokens) == len(sentence_labels)
                    if len(sentence_tokens) == 0:
                        # Happens after a paragraph change (there are 2 blank lines)
                        continue

                    sentences.append({
                        "tokens": sentence_tokens,
                        "labels": sentence_labels
                    })
                    sentence_tokens = []
                    sentence_labels = []
                else:
                    sentence_tokens.append(line[token_position])
                    sentence_labels.append(line[label_position])

        if self.label2id is None:
            # To get all the available labels we have to traverse each list of
            # labels (remember there is a list of labels per data point, not a
            # single label)
            self.label2id = {
                lbl: idx for idx, lbl in
                enumerate(  # Get label and index
                    sorted(  # From the sorted list
                        set(  # Of the set of labels (unique per label)
                            chain(*[  # Concatenate the lists of labels in each sentence
                                sentence["labels"] for sentence in sentences
                            ])
                        )
                    )
                )
            }

        self.dataset = [
            {
                "tokens": sentence['tokens'],
                "labels": [self.label2id[lbl] for lbl in sentence['labels']]
            } for sentence in sentences
        ]

    def _tokenize_and_align_labels(self,
                                   sentence: Dict[str, Union[List[str], List[int]]])\
            -> BatchEncoding:
        """
        Function to tokenize and align labels. This is needed since transformers
        Tokenizer (which use subword tokenization) can split a single token in
        multiple ones, thus breaking the symmetry between tokens and labels.

        Adapted from the code at:
        https://huggingface.co/docs/transformers/tasks/token_classification#preprocess

        Parameters
        ==========
        sentence : Dict[str, List[str] | List[int]]
            A sentence from the `dataset` attribute that is comprised of a
            dictionary that maps two keys ('tokens' and 'labels') to a list of
            tokens and labels.

        Returns
        =======
        Dict[str, List[int]]
            The tokenized sentence as a dictionary with the corresponding key,
            value pairs needed to be processes by a Transformer Module.
        """
        tokenized_sentence = self.tokenizer(
            sentence["tokens"],
            padding='max_length' if self.max_seq_length else False,
            truncation=self.truncation_strategy,
            max_length=self.max_seq_length,
            is_split_into_words=True
        )

        sentence_labels = []
        word_ids = tokenized_sentence.word_ids()
        previous_wid = None
        for wid, attmsk in zip(word_ids, tokenized_sentence['attention_mask']):
            if wid is None:
                if attmsk == 1:
                    # For special tokens ([CLS], [SEP], <s>, etc) that don't have an assigned label,
                    # depending on the configuration, we use the extension label 'X' or
                    # the 'PAD' label
                    extension_label = self.label2id['X'] if self.use_extension_label\
                        else self.label2id['PAD']
                    sentence_labels.append(extension_label)
                if attmsk == 0:
                    # This is a token for padding, assign the PAD label
                    sentence_labels.append(self.label2id['PAD'])
            elif wid != previous_wid:
                # If it is the first subtoken of a work, use its corresponding label
                sentence_labels.append(sentence["labels"][wid])
            else:
                # The other subtokens depend on configuration. For some cases it can be
                # to replicate the label of the first subtoken. If not, the value depends
                # on whether there is an extension label 'X' or the 'PAD' label
                if self.copy_label_to_subtoken:
                    # WARNING: This duplicates the "B-" type labels, but using another
                    # configuration to avoid that is outside this scope since it needs
                    # comparable experimentation as to wether it is useful or not
                    sentence_labels.append(sentence["labels"][wid])
                elif self.use_extension_label:
                    sentence_labels.append(self.label2id['X'])
                else:
                    sentence_labels.append(self.label2id['PAD'])
            previous_wid = wid

        tokenized_sentence['labels'] = sentence_labels

        assert len(tokenized_sentence['input_ids']) == len(tokenized_sentence['labels'])

        return tokenized_sentence

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> BatchEncoding:
        if isinstance(idx, slice):
            # Tokenize each sentence separately
            tokenized_sentences = [self._tokenize_and_align_labels(sentence)
                                   for sentence in self.dataset[idx]]
            # Build a batch encoding from them:
            # The following snippet takes each of the possible keys (from the
            # list of dictionaries that is tokenized_sentences) and
            # builds a dictionary with each of the possible keys in the set
            # of keys. Each value of the dictionary is a list
            # of the values for such key in all the dictionaries
            # Thus: [{k1: va1, k2: va2}, {k1: vb1, k2: vb2}] becomes
            # {k1: [va1, vb1], k2: [va2, vb2]}
            # Since the values are all lists, this returns a dictionary
            # with the values being a list of lists
            return BatchEncoding({
                key: [ts.get(key, []) for ts in tokenized_sentences]
                for key in set(chain(*[ts.keys() for ts in tokenized_sentences]))
            })
        else:
            return self._tokenize_and_align_labels(self.dataset[idx])


class SequenceTaggingDataModule(BaseDataModule):
    """
    DataModule for sequence tagging (i.e. classify each token in a sequence of
    tokens).
    """
    LABELS = [
        "PAD",
        "O",
        "B-Claim",
        "I-Claim",
        "B-Premise",
        "I-Premise",
        "B-Marker",
        "I-Marker",
        "B-Treatment",
        "I-Treatment",
        "B-Disease",
        "I-Disease",
        "B-Diagnostics",
        "I-Diagnostics"
    ]

    @property
    def collate_fn(self) -> Callable:
        return DataCollatorForTokenClassification(self.tokenizer,
                                                  label_pad_token_id=self.label2id['PAD'])

    def decode_predictions(self,
                           input_ids: Union[List[int], List[List[int]]],
                           predictions: Union[List[int], List[List[int]]],
                           labels: Union[List[int], List[List[int]], None] = None)\
            -> Union[List[Tuple[str]], List[List[Tuple[str]]]]:
        """
        Decodes the input_ids, which can be a single instance (List[int]) or a
        batch of instances (List[List[int]]) into the corresponding list of
        tokens with their associated prediction (one per token). If true
        labels are provided, it will add them as well.

        TODO (Issue #10): This method doesn't make any realignment of tokens and
        labels, it doesn't regroup the sub-tokens and simply returns each
        subtoken with the corresponding label of the subtoken. For the next
        iteration we need to address this.

        Parameters
        ==========
        input_ids: List[int] | List[List[int]]
            The tokens ids of a single instance or a batch of instances.
        predictions: List[int] | List[List[int]]
            The predicted labels for a single instance or the list of predicted
            labels for a batch of instances.
        labels: List[int] | List[List[int]], optional
            If given, the true labels.

        Return
        ======
        List[Tuple[str]], List[List[Tuple[str]]]
            Depending on the type of input, it can return the predictions of a
            single input (List[Tuple[str]]) or for a batch of inputs
            (List[List[Tuple[str]]]). The tuple has the form: (token,
            predicted_label) or (token, predicted_label, true_label) if the true
            label is given.
        """
        if isinstance(input_ids[0], int):
            special_tokens = self.tokenizer.get_special_tokens_mask(input_ids,
                                                                    already_has_special_tokens=True)
            if labels is None:
                return [
                    (
                        self.tokenizer.convert_ids_to_tokens(token),
                        self.id2label[prediction]
                    )
                    for token, prediction, mask in zip(input_ids, predictions, special_tokens)
                    if mask == 0
                ]
            else:
                return [
                    (
                        self.tokenizer.convert_ids_to_tokens(token),
                        self.id2label[prediction],
                        self.id2label[label]
                    )
                    for token, prediction, label, mask in zip(input_ids, predictions,
                                                              labels, special_tokens)
                    if mask == 0
                ]
        else:
            outputs = []
            if labels is None:
                for stokens, spredictions in zip(input_ids, predictions):
                    special_tokens = self.tokenizer.get_special_tokens_mask(
                        stokens, already_has_special_tokens=True
                    )
                    outputs.append([(
                            self.tokenizer.convert_ids_to_tokens(token),
                            self.id2label[prediction]
                        ) for token, prediction, mask in zip(stokens, spredictions,
                                                             special_tokens)
                          if mask == 0
                    ])
            else:
                for stokens, spredictions, slabels in zip(input_ids, predictions, labels):
                    special_tokens = self.tokenizer.get_special_tokens_mask(
                        stokens, already_has_special_tokens=True
                    )
                    outputs.append([(
                            self.tokenizer.convert_ids_to_tokens(token),
                            self.id2label[prediction],
                            self.id2label[label]
                        ) for token, prediction, label, mask in zip(stokens, spredictions, slabels,
                                                                    special_tokens)
                          if mask == 0
                    ])
            return outputs

    def _load_dataset_split(self, path_to_dataset: str) -> SequenceTaggingDataset:
        """
        Check BaseDataModule for documentation.
        """
        return SequenceTaggingDataset(
            tokenizer=self.tokenizer,
            path_to_dataset=path_to_dataset,
            label2id=self.labels,
            **self.datasets_config
        )

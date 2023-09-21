from itertools import chain
from transformers.tokenization_utils_base import BatchEncoding
from typing import Dict, List, Optional, Union

from .base import BaseDataset


class SequenceTaggingDataset(BaseDataset):
    """
    Dataset for sequence tagging (i.e. classify each token in a sequence of
    tokens).

    It expects data in CONLL format, where each token is associated to a "BIO"
    style label.

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
    token_position: int
        The position (column, with start on 0) where the token is present in the
        CONLL dataset.
    label_position: int
        The position (column, with start on 0) where the label is present in the
        CONLL dataset.
    use_extension_label: bool
        If true, uses a special label 'X' for special tokens in the transformer
        (e.g. [CLS], [SEP], <s>, etc.). If false, it will replace such labels
        with a special padding label 'PAD' equal to -100 that will be ignored by
        losses such as `torch.nn.CrossEntropy`. Careful because losses such as
        `pytorchcrf.CRF` will most likely fail (or have an unexpected behavior)
        for the case of using the -100 label.
        For more information on why to use -100 check:
        https://huggingface.co/docs/transformers/tasks/token_classification#preprocess
    copy_label_to_subtoken: bool
        If true, when a token is separated in subtokens by the tokenizer, it
        copies the label to each subtoken. If it false, the assigned label to
        each subtoken after the first will depend on the value of
        `use_extension_label`. If the extension label 'X' exists, it will use
        it, otherwise it will use the special 'PAD' label assigned to -100.
    """
    def __init__(self,
                 tokenizer_model_or_path: str,
                 path_to_dataset: str,
                 labels: Optional[Dict[str, int]] = None,
                 delimiter: str = '\t',
                 token_position: int = 1,
                 label_position: int = 4,
                 use_extension_label: bool = True,
                 copy_label_to_subtoken: bool = True):
        super().__init__(tokenizer_model_or_path=tokenizer_model_or_path,
                         path_to_dataset=path_to_dataset, labels=labels,
                         delimiter=delimiter, token_position=token_position,
                         label_position=label_position)

        self.use_extension_label = use_extension_label
        self.copy_label_to_subtoken = copy_label_to_subtoken
        if self.use_extension_label:
            if 'X' not in self.labels:
                # Add the extended label
                self.labels['X'] = len(self.labels)
            if 'PAD' not in self.labels:
                # Add the pad label (after the extension label)
                self.labels['PAD'] = len(self.labels)
        elif 'PAD' not in self.labels:
            # The pad label will be -100
            self.labels['PAD'] = -100

    def _load_dataset(self,
                      path_to_dataset: str,
                      missing_labels: bool = False,
                      delimiter: str = '\t',
                      token_position: str = 1,
                      label_position: str = 4):
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

        if missing_labels:
            # To get all the available labels we have to traverse each list of
            # labels (remember there is a list of labels per data point, not a
            # single label)
            self.labels = {
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
                "labels": [self.labels[lbl] for lbl in sentence['labels']]
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
        ----------
        sentence : Dict[str, List[str] | List[int]]
            A sentence from the `dataset` attribute that is comprised of a
            dictionary that maps two keys ('tokens' and 'labels') to a list of
            tokens and labels.

        Returns
        -------
        Dict[str, List[int]]
            The tokenized sentence as a dictionary with the corresponding key,
            value pairs needed to be processes by a Transformer Module.
        """
        tokenized_sentence = self.tokenizer(sentence["tokens"], truncation=True,
                                            is_split_into_words=True)

        sentence_labels = []
        word_ids = tokenized_sentence.word_ids()
        previous_wid = None
        for wid in word_ids:
            if wid is None:
                # For special tokens ([CLS], [SEP], <s>, etc) that don't have an assigned label
                # we use a special extension_label that, dependind on configuration can be a value
                # for a extension label X or a special PAD label equal to -100 (in case of using
                # regular transformers and not CRF which requires all labels to be accounted).
                extension_label = self.labels['X'] if self.use_extension_label\
                    else self.labels['PAD']
                sentence_labels.append(extension_label)
            elif wid != previous_wid:
                # If it is the first subtoken of a work, use its corresponding label
                sentence_labels.append(sentence["labels"][wid])
            else:
                # The other subtokens depend on configuration. For some cases it can be
                # to replicate the label of the first subtoken. If not, the value depends
                # on whether there is an extension label X or plainly padding with -100
                if self.copy_label_to_subtoken:
                    # WARNING: This duplicates the "B-" type labels, but using another
                    # configuration to avoid that is outside this scope since it needs
                    # comparable experimentation as to wether it is useful or not
                    sentence_labels.append(sentence["labels"][wid])
                elif self.use_extension_label:
                    sentence_labels.append(self.labels['X'])
                else:
                    sentence_labels.append(self.labels['PAD'])
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

"""
The pipeline module hold different functions to run a full annotation pipeline.

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

from collections import defaultdict
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    # Only import this modules if needed for type checking
    # This is required to follow pep8 on type hinting and annotations
    from acta.models import RelationClassificationTransformerModule, \
        SequenceTaggingTransformerModule

logger = logging.getLogger(__name__)


def relation_classification(text: List[str],
                            text_pair: List[str],
                            model: "RelationClassificationTransformerModule",
                            tokenizer: Optional[Union[str, AutoTokenizer]] = None,
                            id2label: Optional[Dict[int, str]] = None,
                            max_seq_lenght: Optional[int] = None,
                            truncation_strategy: Optional[Union[str, bool]] = False) -> List[str]:
    """
    Function to do relationship classification between pairs of argumentation
    components. The components should already have been obtained from a previous
    iteration (i.e. this function doesn't detect components nor does any kind of
    sequence tagging, for that refer to the `sequence_tagging` function). It also
    doesn't make any assumptions on the input text.

    Parameters
    ----------
    text: List[str]
        A list with the sentences in the first group (after the [CLS] but before
        the first [SEP]).
    text_pair: List[str]
        A list with the sentences in the second group (between the first [SEP]
        and the final [SEP]). It is expected that text_pair[i] is the sentence
        corresponding to text[i] since the relation classification will be done
        with that assumption.
    model: RelationClassificationTransformerModule
        The model to run the classification. It might be as well any model that
        conforms with HF AutoModelForSequenceClassification.
    tokenizer: str | AutoTokenizer, optional
        The tokenizer to process the data (either as a string matching to some
        tokenizer in Hugging Face hub) or as a pre-trained tokenizer. If it's
        not given it will use the fast tokenizer that is associated by default
        to the model.
    id2label: Dict[int, str], optional
        The mapping between labels ids and final labels in the 'BIO' format. It
        is important that it is in that format since the heuristics will work
        only with that format. If not given it will try to get it from the
        model, assuming the model is actually a
        RelationClassificationTransformerModule.
    max_seq_length: int, optional
        If > 0 truncates and pads each sequence of the dataset to `max_seq_length`.
    truncation_strategy: str|bool, optional
        What truncation strategy to use. Must be one of `longest_first` (or
        True), `only_second`, `only_first` or 'do_not_truncate' (or False).
        Check https://huggingface.co/docs/transformers/pad_truncation for more
        information.

    Returns
    -------
    List[str]
        The list of labels indicating the relation type.
    """
    if tokenizer is None:
        # When tokenizer is missing, use the model tokenizer with a warning
        logger.warning("There was no tokenizer information given, defaulting "
                       "to the tokenizer with the same name of the model: "
                       f"{model.config.name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path, use_fast=True)
    elif isinstance(tokenizer, str):
        logger.info(f"Loading tokenizer: {tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)

    if id2label is None:
        logger.warning("The `id2label` is taken from the model configuration.")
        id2label = model.config.id2label

    tokenized_components = tokenizer(text=text, text_pair=text_pair, padding=True,
                                     max_length=max_seq_lenght, truncation=truncation_strategy,
                                     return_tensors='pt')
    predictions = model(**tokenized_components).logits.argmax(1).tolist()

    return [id2label[pred] for pred in predictions]


def _sentence_annotation(tokens: List[int],
                         predictions: List[int],
                         special_tokens_mask: List[int],
                         tokenizer: AutoTokenizer,
                         id2label: Dict[int, str]) -> List[Dict[str, str]]:
    """
    Auxiliary function to annotate a single sentence (as a sequence of tokens),
    used by the `sequence_tagging` function.

    Parameters
    ----------
    tokens: List[int]
        The list of token ids.
    predictions: List[int]
        The list of predicted labels.
    special_tokens_mask: List[int]
        The mask for special tokens.
    tokenizer: AutoTokenizer
        The tokenizer (needed for decoding the token ids to text).
    id2label: Dict[int, str]
        The mapping between labels ids and 'BIO' labels.

    Returns
    -------
    List[Dict[str, str]]
        The annotated groups for the sentence.
    """
    assert len(tokens) == len(predictions) == len(special_tokens_mask), \
        "There was a problem when using the model for inference. " \
        "The tokens and predictions sizes do not match."
    annotations = []
    current_annotation = {}

    for i, (token, prediction, mask) in enumerate(zip(tokens, predictions, special_tokens_mask)):
        label = id2label[prediction]
        if mask == 1:
            # We ignore special tokens from the final result
            continue
        if not current_annotation:
            # There isn't an annotation group yet, start one, regardless of label type.
            # We strip the 'B-' and 'I-' from the labels start to aggregate the annotations
            current_annotation = {"label": label.lstrip('B-').lstrip('I-'), "tokens": [token]}
        elif label.startswith('B-'):
            # This annotation heuristics assumes that the 'B-' label is generally correct, and
            # tries to start a new annotation group unless the following condition is met
            if len(current_annotation['tokens']) < 3 and current_annotation['label'] != 'O':
                # If the current label starts with a 'B-' but there are less than 3 tokens in
                # the current annotation with the current label being other than the 'O' label
                # then we have a token that is part of the current annotation.
                # This happens when having something like 'B-', 'O', 'B-', 'I-'...
                current_annotation['tokens'].append(token)
            else:
                # If there are 3 or more tokens already in the current annotation
                # or the current label is 'O' we start a new annotation
                # We decode the tokens into a text (removing them), append the
                # current annotation to the list of annotations and reset the
                # current annotation
                current_annotation['text'] = tokenizer.decode(current_annotation.pop('tokens'))
                annotations.append(current_annotation)
                current_annotation = {"label": label.lstrip('B-'), "tokens": [token]}
        elif label.startswith('I-'):
            # For the 'I-' labels we only create a new annotation if we are coming
            # from the 'O' annotation. This is for the weird cases where there are
            # no B- labels within reach.
            if current_annotation['label'] == 'O':
                current_annotation['text'] = tokenizer.decode(current_annotation.pop('tokens'))
                annotations.append(current_annotation)
                current_annotation = {"label": label.lstrip('I-'), "tokens": [token]}
            else:
                # When coming from another label we just add the token to the group.
                # This heuristics assumes that the current annotation has the right label
                # which was started by a 'B-' type label (or the weird cases with a 'I-' type
                # label contemplated in the previous condition).
                # The 'I-' labels don't have precedence over the 'B-' labels to decide the final
                # label. Thus if we have 'B-lbl1', 'I-lbl2', 'I-lbl2', the final label
                # will be 'label1'. This is a simple heuristics, in observations the
                # 'B-' type label is the correct one, and many times it can happen something like
                # 'B-lbl1', 'I-lbl1', 'I-lbl2', 'I-lbl1', 'I-lbl1'
                current_annotation['tokens'].append(token)
        else:
            # For the 'O' labels, it can be a fluke contained between 'B-'/'I-' labels.
            # We check the next 2 labels (if available) and decide what to do based on that.
            if i + 1 < len(predictions) and id2label[predictions[i+1]] != 'O':
                # It's a fluke, treat it as part of the current annotation
                current_annotation['tokens'].append(token)
            elif i + 2 < len(predictions) and id2label[predictions[i+2]] != 'O':
                # It's a fluke, treat it as part of the current annotation
                current_annotation['tokens'].append(token)
            elif i + 1 == len(predictions):
                # We are in the last token, we always assume is part of the current annotation
                current_annotation['tokens'].append(token)
            elif current_annotation['label'] == label:
                # If the current annotation has label 'O' already, then it's
                # part of the current annotation
                current_annotation['tokens'].append(token)
            else:
                # We have more than 2 occurrences of the 'O' label (or we are at the tail end
                # of the text), we can assume that we have a group of 'O' annotated tokens
                # thus we start a new annotation for them
                current_annotation['text'] = tokenizer.decode(current_annotation.pop('tokens'))
                annotations.append(current_annotation)
                current_annotation = {"label": label, "tokens": [token]}

    if current_annotation.get('tokens'):
        # After traversing all the tokens, we check if we still have an unprocessed
        # annotation (i.e. it has a list of tokens that hasn't been decoded).
        # If so, we decode them and add them to the annotations
        current_annotation['text'] = tokenizer.decode(current_annotation.pop('tokens'))
        annotations.append(current_annotation)

    return annotations


def sequence_tagging(text: str,
                     model: "SequenceTaggingTransformerModule",
                     tokenizer: Optional[Union[str, AutoTokenizer]] = None,
                     id2label: Optional[Dict[int, str]] = None) -> List[Dict[str, str]]:
    """
    Function to do token classification annotation. The function expects a text
    and a SequenceTaggingTransformerModule (already trained) and based on the
    predictions over the text sequence, it will try to aggregate all parts of
    the text that conform as having the same label.

    Since the models rarely perform without any error or noise, the module does
    some heuristics over the predicted labels to avoid having many short lived
    spans of varying labels:
    E.g. If the function encounters something like this:
        B-lbl, O, B-lbl, I-lbl, ...
        It will ignore the 'O' and make it as part of the 'lbl' in the group.

    Also, the 'B-' type labels take precedence over the 'I-' type labels.
    E.g. If the model finds:
        B-lbl1, I-lbl2, I-lbl2, I-lbl1, ...
        It will set the group as having the 'lbl1' instead of 'lbl2'

    Check the comments on the code for further understanding of other
    heuristics.

    Parameters
    ----------
    text: str
        The text to be annotated. It will be splitted into sentences with NLTK
        and then will be tokenized and processed with the corresponding HF
        tokenizer.
    model: SequenceTaggingTransformerModule
        The model to use for the annotations. It is expected this module to be
        able to do sequence tagging. It is expected to be a SequenceTaggingTransformerModule
        but it's technically possible to use either PyTorch based module that
        defines it's `forward` method returning a tuple/list where the first element
        is the tensor with the predictions.
    tokenizer: str | AutoTokenizer, optional
        The tokenizer to process the data (either as a string matching to some
        tokenizer in Hugging Face hub) or as a pre-trained tokenizer. If it's
        not given it will use the fast tokenizer that is associated by default
        to the model.
    id2label: Dict[int, str], optional
        The mapping between labels ids and final labels in the 'BIO' format. It
        is important that it is in that format since the heuristics will work
        only with that format. If not given it will try to get it from the
        model, assuming the model is actually a
        SequenceTaggingTransformerModule.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries with the aggregated annotations. It has one
        dictionary per group of annotations and it follows the order encountered
        in the original text. Each dictionary is comprised of 2 elements:
            - The 'label': A string with the label of the annotated group
              (without the 'B-'/'I-' prepends, it aggregates them)
            - The 'text': The text of the annotated group (as processed and
              decoded by the tokenizer).
    """
    if tokenizer is None:
        # When tokenizer is missing, use the model tokenizer with a warning
        logger.warning("There was no tokenizer information given, defaulting "
                       "to the tokenizer with the same name of the model: "
                       f"{model.config.name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path, use_fast=True)
    elif isinstance(tokenizer, str):
        logger.info(f"Loading tokenizer: {tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)

    if id2label is None:
        logger.warning("The `id2label` is taken from the model configuration.")
        id2label = model.config.id2label

    # Any label that is not part of the BIO tags, will default to O. This is to
    # avoid problem with 'PAD' or 'X' tags (i.e. for padding and extensions)
    id2label = defaultdict(
        lambda: 'O',
        {idx: lbl for idx, lbl in id2label.items() if lbl[0] in 'BIO'}
    )

    sentences = sent_tokenize(text)
    tokenized_text = tokenizer(sentences, padding=True, return_tensors='pt',
                               return_special_tokens_mask=True)
    special_tokens_mask = tokenized_text.pop('special_tokens_mask')
    predictions = model(**tokenized_text)

    annotations = []

    for sentence_idx in range(tokenized_text['input_ids'].size(0)):
        # For each sentence, run the annotation function for a single sentence
        sentence_tokens = tokenized_text['input_ids'][sentence_idx].tolist()
        sentence_predictions = predictions[0][sentence_idx].tolist()
        sentence_special_tokens_mask = special_tokens_mask[sentence_idx].tolist()
        sentence_annotations = _sentence_annotation(
            tokens=sentence_tokens,
            predictions=sentence_predictions,
            special_tokens_mask=sentence_special_tokens_mask,
            tokenizer=tokenizer,
            id2label=id2label
        )
        annotations.extend(sentence_annotations)

    return annotations

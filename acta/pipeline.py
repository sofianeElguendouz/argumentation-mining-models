"""
The pipeline module hold different functions to run a full annotation pipeline.

    Argumentation Mining Transformers Pipeline Utilities Module
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

import logging

from collections import defaultdict
from itertools import groupby
from nltk.tokenize import sent_tokenize
from torch import softmax
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

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
                            max_seq_length: Optional[int] = None,
                            truncation_strategy: str = 'do_not_truncate',
                            return_confidence: bool = False,
                            confidence_as_probability: bool = False) \
                                -> List[Dict[str, Union[str, float]]]:
    """
    Function to do relationship classification between pairs of argumentation
    components. The components should already have been obtained from a previous
    iteration (i.e. this function doesn't detect components nor does any kind of
    sequence tagging, for that refer to the `sequence_tagging` function). It also
    doesn't make any assumptions on the input text.

    Parameters
    ==========
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
        Only makes sense if `truncation_strategy` != 'do_not_truncate'.
    truncation_strategy: str
        What truncation strategy to use. Must be one of `longest_first`,
        `only_second`, `only_first` or `do_not_truncate`.
        Check https://huggingface.co/docs/transformers/pad_truncation for more
        information.
    return_confidence: bool
        If True, return confidence (logits) alongside predictions.
    confidence_as_probability: bool
        If True, transforms the confidence in probability with a SoftMax
        function.

    Returns
    =======
    List[Dict[str, str | float]]
        A list with dictionary with the prediction for each pair of elements
        given by parameter. Each dictionary has the following elements:
            - The 'label': The label (mapped from id2label) of the prediction.
            - The 'confidence': This is optional (depends on
              `return_confidence`) and it can be either the logit score of the
              prediction or the probability (if `confidence_as_probability` is
              True).
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
                                     max_length=max_seq_length, truncation=truncation_strategy,
                                     return_tensors='pt')
    scores = model(**tokenized_components).logits
    if confidence_as_probability:
        scores = softmax(scores, 1)
    confidence = scores.max(1).values.tolist()
    predictions = scores.argmax(1).tolist()

    if return_confidence:
        return [
            {
                'label': id2label[prediction],
                'confidence': confidence
            } for prediction, confidence in zip(predictions, confidence)
        ]
    else:
        return [
            {'label': id2label[prediction]} for prediction in predictions
        ]


def _sentence_annotation(tokens: List[int],
                         predictions: List[int],
                         word_ids: List[int],
                         tokenizer: AutoTokenizer,
                         id2label: Dict[int, str]) -> List[Dict[str, str]]:
    """
    Auxiliary function to annotate a single sentence (as a sequence of tokens),
    used by the `sequence_tagging` function.

    Parameters
    ==========
    tokens: List[int]
        The list of token ids.
    predictions: List[int]
        The list of predicted labels.
    word_ids: List[int]
        The word ids that identify what word are part of each of the subtokens.
    tokenizer: AutoTokenizer
        The tokenizer (needed for decoding the token ids to text).
    id2label: Dict[int, str]
        The mapping between labels ids and 'BIO' labels.

    Returns
    =======
    List[Dict[str, str]]
        The annotated groups for the sentence.
    """
    assert len(tokens) == len(predictions) == len(word_ids), \
        "There was a problem when using the model for inference. " \
        "The tokens and predictions sizes do not match."
    annotations = []
    current_annotation = {}

    for i, (token, prediction, wid) in enumerate(zip(tokens, predictions, word_ids)):
        label = id2label[prediction]
        if wid is None:
            # We ignore special tokens from the final result
            continue
        if not current_annotation:
            # There isn't an annotation group yet, start one, regardless of label type.
            # We strip the 'B-' and 'I-' from the labels start to aggregate the annotations
            current_annotation = {
                "label": label.removeprefix('B-').removeprefix('I-'),
                "tokens": [token],
                "word_ids": [wid]
            }
        elif wid == current_annotation["word_ids"][-1]:
            # If the current token is a continuation of the previous one we have a couple of
            # possibilities
            if label.removeprefix('B-').removeprefix('I-') == current_annotation['label']\
                    or label == 'O':
                # If the subtoken has the same group label as the current annotation, or
                # if the current subtoken is annotated with and 'O', make it part of the current
                # annotation, avoiding cases with a word splitted in multiple subtokens and
                # each subtoken being part of a different label
                current_annotation['tokens'].append(token)
                current_annotation['word_ids'].append(wid)
            else:
                # If the subtoken has a label different than the current annotation, but is not the
                # 'O' label (e.g. we have a subtoken with a 'B-' or 'I-' type of label) then
                # give that label precedence by starting a new annotation with that label and all
                # the subtokens that are part of the token
                tokens = current_annotation.pop('tokens')
                wids = current_annotation.pop('word_ids')

                # Get the subtokens that are not part of the token that contains the current
                # subtoken with different label
                current_annotation_tokens = [t for t, w in zip(tokens, wids) if w != wid]
                current_annotation['text'] = tokenizer.decode(current_annotation_tokens)
                annotations.append(current_annotation)

                # Create a new annotation with the current subtoken's token
                current_annotation = {
                    "label": label.removeprefix('B-').removeprefix('I-'),
                    "tokens": [t for t, w in zip(tokens, wids) if w == wid],
                    "word_ids": [w for w in wids if w == wid]
                }
        elif label.startswith('B-'):
            # This annotation heuristics assumes that the 'B-' label is generally correct, and
            # tries to start a new annotation group except for some conditions
            if len(current_annotation['tokens']) < 3 and current_annotation['label'] != 'O':
                # If the current label starts with a 'B-' but there are less than 3 tokens in
                # the current annotation with the current label being other than the 'O' label
                # then we have a token that is part of the current annotation.
                # This happens when having something like 'B-', 'O', 'B-', 'I-'...
                current_annotation['tokens'].append(token)
                current_annotation['word_ids'].append(wid)
            else:
                # If there are 3 or more tokens already in the current annotation
                # or the current label is 'O' we start a new annotation
                # We decode the tokens into a text (removing them), append the
                # current annotation to the list of annotations and reset the
                # current annotation
                current_annotation['text'] = tokenizer.decode(current_annotation.pop('tokens'))
                annotations.append(current_annotation)
                current_annotation = {
                    "label": label.removeprefix('B-'),
                    "tokens": [token],
                    "word_ids": [wid]
                }
        elif label.startswith('I-'):
            # For the 'I-' labels we only create a new annotation if we are coming
            # from the 'O' annotation. This is for the weird cases where there are
            # no B- labels within reach.
            if current_annotation['label'] == 'O':
                current_annotation['text'] = tokenizer.decode(current_annotation.pop('tokens'))
                annotations.append(current_annotation)
                current_annotation = {
                    "label": label.removeprefix('I-'),
                    "tokens": [token],
                    "word_ids": [wid]
                }
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
                current_annotation['word_ids'].append(wid)
        else:
            # For the 'O' labels, it can be a fluke contained between 'B-'/'I-' labels.
            # We check the next 2 labels (if available) and decide what to do based on that.
            if i + 1 < len(predictions) and id2label[predictions[i+1]] != 'O':
                # It's a fluke, treat it as part of the current annotation
                current_annotation['tokens'].append(token)
                current_annotation['word_ids'].append(wid)
            elif i + 2 < len(predictions) and id2label[predictions[i+2]] != 'O':
                # It's a fluke, treat it as part of the current annotation
                current_annotation['tokens'].append(token)
                current_annotation['word_ids'].append(wid)
            elif i + 1 == len(predictions):
                # We are in the last token, we always assume is part of the current annotation
                current_annotation['tokens'].append(token)
                current_annotation['word_ids'].append(wid)
            elif current_annotation['label'] == label:
                # If the current annotation has label 'O' already, then it's
                # part of the current annotation
                current_annotation['tokens'].append(token)
                current_annotation['word_ids'].append(wid)
            else:
                # We have more than 2 occurrences of the 'O' label (or we are at the tail end
                # of the text), we can assume that we have a group of 'O' annotated tokens
                # thus we start a new annotation for them
                current_annotation['text'] = tokenizer.decode(current_annotation.pop('tokens'))
                annotations.append(current_annotation)
                current_annotation = {
                    "label": label,
                    "tokens": [token],
                    "word_ids": [wid]
                }

    if current_annotation.get('tokens'):
        # After traversing all the tokens, we check if we still have an unprocessed
        # annotation (i.e. it has a list of tokens that hasn't been decoded).
        # If so, we decode them and add them to the annotations
        current_annotation['text'] = tokenizer.decode(current_annotation.pop('tokens'))
        current_annotation.pop('word_ids')  # Remove the word_ids key since it's no longer needed
        annotations.append(current_annotation)

    return annotations


def sequence_tagging(text: str,
                     model: "SequenceTaggingTransformerModule",
                     tokenizer: Optional[Union[str, AutoTokenizer]] = None,
                     id2label: Optional[Dict[int, str]] = None,
                     max_seq_length: Optional[int] = None,
                     truncation_strategy: str = 'do_not_truncate') \
                        -> Tuple[List[Dict[str, str]], List[int]]:
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

    Finally, the heuristics always make all the subtokens conforming a single
    token part of the same label. And if one subtoken has a B/I label, it will
    take precedence over any 'O' label the other subtokens of the token have.

    Check the comments on the code for further understanding of other
    heuristics.

    Parameters
    ==========
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
    max_seq_length: int, optional
        If > 0 truncates and pads each sequence of the dataset to `max_seq_length`.
        Only makes sense if `truncation_strategy` != 'do_not_truncate'.
    truncation_strategy: str
        What truncation strategy to use. Must be one of `longest_first`,
        `only_second`, `only_first` or `do_not_truncate`.
        Check https://huggingface.co/docs/transformers/pad_truncation for more
        information.

    Returns
    =======
    Tuple[List[Dict[str, str]], List[int]]
        The returned value is a tuple where:
        - The first is a list of dictionaries with the aggregated annotations. It
          has one dictionary per group of annotations and it follows the order
          encountered in the original text. Each dictionary is comprised of 3
          elements:
            - The 'sidx': An integer representing the index of the sentence in
              the whole text (which is splitted into sentences for easier
              processing). This serves for reconstructing the sentences
              afterwise.
            - The 'label': A string with the label of the annotated group
              (without the 'B-'/'I-' prepends, it aggregates them)
            - The 'text': The text of the annotated group (as processed and
              decoded by the tokenizer).
        - The second is a list with the indices of the relevant annotations,
          i.e. the indices to those annotations with a label that is different
          from the 'O' label.
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
                               max_length=max_seq_length, truncation=truncation_strategy)
    predictions = model(**tokenized_text)

    annotations = []

    for sentence_idx in range(tokenized_text['input_ids'].size(0)):
        # For each sentence, run the annotation function for a single sentence
        sentence_tokens = tokenized_text['input_ids'][sentence_idx].tolist()
        sentence_predictions = predictions[0][sentence_idx].tolist()
        sentence_word_ids = tokenized_text.word_ids(sentence_idx)
        sentence_annotations = _sentence_annotation(
            tokens=sentence_tokens,
            predictions=sentence_predictions,
            word_ids=sentence_word_ids,
            tokenizer=tokenizer,
            id2label=id2label
        )
        # Add the sentence index to each of the sentences as we append them to the annotations
        annotations.extend([
            {'sidx': sentence_idx, **sentence} for sentence in sentence_annotations
        ])

    # As a final step, to avoid having lots of irrelevant elements, we group all
    # annotations that have the 'O' label and are contiguous in the same sentence.
    # This is only for the 'O' label since contiguous relevant labels can be very well different
    # elements.
    grouped_annotations = []
    for (label, sidx), group in groupby(annotations, key=lambda ann: (ann['label'], ann['sidx'])):
        if label == 'O':
            # We aggregate the contiguous 'O' annotations that are part of the same sentence
            grouped_annotations.append({
                'sidx': sidx,
                'label': label,
                'text': ' '.join([ann['text'].strip() for ann in group])
            })
        else:
            # We leave the rest as is
            grouped_annotations.extend(group)

    relevant = [
        idx for idx, annotation in enumerate(grouped_annotations) if annotation['label'] != 'O'
    ]

    return grouped_annotations, relevant

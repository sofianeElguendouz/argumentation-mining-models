"""
This module holds different types of utilities (functions and classes) for the
Argumentation Mining Transformers library.

    Argumentation Mining Transformers Utilities Module
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

from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Optional, Union


def compute_metrics(true_labels: List[Union[int, str]],
                    pred_labels: List[Union[int, str]],
                    relevant_labels: Optional[List[Union[int, str]]] = None,
                    prefix: Optional[str] = None) -> Dict[str, float]:
    """
    Function to compute accuracy and F1-score, macro and micro averaged, with the
    option to limit the F1-score only to relevant labels.

    Parameters
    ==========
    true_labels: List[int|str]
        The list (or can be any ordered iterable as well) of the true labels
        (ground truth).
    pred_labels: List[int|str]
        The list (or can be any ordered iterable as well) of the predicted
        labels.
    relevant_labels: Optional[List[int|str]]
        If given, it will calculate the F1-score micro and macro averaged only
        over the given labels (relevant ones). Useful for example to remove the
        most common labels from the final metrics.
    prefix: Optional[str]
        If given will prepend `<prefix>_` to the name of the metric. Useful to
        check, for example, what was the split the metric was done in (e.g.
        `eval_accuracy`).

    Returns
    =======
    Dict[str, float]
        A mapping between each of the metrics names (optionally with the
        prepended prefix) and the metric value.
    """
    outputs = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1_score_macro": f1_score(true_labels, pred_labels, average="macro", zero_division=0),
        "f1_score_micro": f1_score(true_labels, pred_labels, average="micro", zero_division=0)
    }

    if relevant_labels is not None:
        # F1 macro and micro averages for specific classes
        outputs["relevant_f1_score_macro"] = f1_score(
            true_labels, pred_labels, average="macro", zero_division=0, labels=relevant_labels
        )
        outputs["relevant_f1_score_micro"] = f1_score(
            true_labels, pred_labels, average="micro", zero_division=0, labels=relevant_labels
        )

    if prefix:
        outputs = {f"{prefix}_{metric}": value for metric, value in outputs.items()}

    return outputs


def compute_seq_tag_labels_metrics(true_labels: List[str],
                                   pred_labels: List[str],
                                   labels: List[str],
                                   prefix: Optional[str] = None) -> Dict[str, float]:
    """
    Function to compute F1-score, macro and micro averaged for groups of labels
    in sequence tagging. This calculates the labels that have the same root
    (i.e. B and I labels of the same class) into the micro and macro F1-score
    average. It will ignore any label that is not B or I type.

    Parameters
    ==========
    true_labels: List[str]
        The list (or can be any ordered iterable as well) of the true labels
        (ground truth). They have to be strings in BIO format.
    pred_labels: List[str]
        The list (or can be any ordered iterable as well) of the predicted
        labels. They have to be strings in BIO format.
    labels: List[str]
        The set of unique labels.
    prefix: Optional[str]
        If given will prepend `<prefix>_` to the name of the metric. Useful to
        check, for example, what was the split the metric was done in (e.g.
        `eval_accuracy`).

    Returns
    =======
    Dict[str, float]
        A mapping between each of the metrics names (optionally with the
        prepended prefix) and the metric value.
    """

    unique_labels = {
        lbl.split('-', 1)[1] for lbl in labels if lbl.startswith('B-') or lbl.startswith('I-')
    }
    labels_groups = {
        lbl: [f'B-{lbl}', f'I-{lbl}'] for lbl in unique_labels
    }

    outputs = {}

    for lbl, lbl_group in labels_groups.items():
        outputs[f'f1_score_macro_{lbl}'] = f1_score(true_labels, pred_labels, labels=lbl_group,
                                                    average='macro', zero_division=0)
        outputs[f'f1_score_micro_{lbl}'] = f1_score(true_labels, pred_labels, labels=lbl_group,
                                                    average='micro', zero_division=0)

    if prefix:
        outputs = {f"{prefix}_{metric}": value for metric, value in outputs.items()}

    return outputs

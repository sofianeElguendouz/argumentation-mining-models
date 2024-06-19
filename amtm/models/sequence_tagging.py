"""
Sequence Tagging Transformer Module for Token Classification in Argumentation Mining.

    Argumentation Mining Transformers Sequence Tagging Transformer Module
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

import torch

from transformers import AutoModelForTokenClassification
from typing import Any, Dict, Optional

from .base import BaseTransformerModule


class SequenceTaggingTransformerModule(BaseTransformerModule):
    """
    Lightning Module for sequence tagging (i.e. classify each token in a
    sequence of tokens).

    Parameters
    ==========
    model_name_or_path: str
        Refer to BaseTransformerModule.
    label2id: Dict[str, int]
        Refer to BaseTransformerModule.
    id2label: Dict[int, str]
        Refer to BaseTransformerModule.
    config_name_or_path: Optional[str]
        Refer to BaseTransformerModule.
    cache_dir: Optional[str]
        Refer to BaseTransformerModule.
    learning_rate: float
        Refer to BaseTransformerModule.
    weight_decay: float
        Refer to BaseTransformerModule.
    adam_epsilon: float
        Refer to BaseTransformerModule.
    warmup_steps: int
        Refer to BaseTransformerModule.
    **kwargs
        Refer to BaseTransformerModule.
    """
    def __init__(self,
                 model_name_or_path: str,
                 label2id: Dict[str, int],
                 id2label: Dict[int, str],
                 config_name_or_path: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.0,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 **kwargs):
        super().__init__(model_name_or_path=model_name_or_path,
                         label2id=label2id, id2label=id2label,
                         config_name_or_path=config_name_or_path,
                         cache_dir=cache_dir, learning_rate=learning_rate,
                         weight_decay=weight_decay, adam_epsilon=adam_epsilon,
                         warmup_steps=warmup_steps, **kwargs)

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            config=self.config,
            cache_dir=cache_dir
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def _loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self(**batch).loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step. It returns the inputs_ids (i.e. tokens ids), the real
        labels (if they are present) and the predictions (which are an argmax
        over the logits).
        """
        labels = batch.pop('labels', None)
        predictions = self(**batch).logits.argmax(dim=-1)

        return {
            "input_ids": batch.input_ids.tolist(),
            "labels": labels.tolist() if labels is not None else None,
            "predictions": predictions.tolist()
        }

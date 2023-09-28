"""
Pytorch Lightning Module for Relation Classfication.

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

import numpy as np
import torch

from transformers import AutoModelForSequenceClassification
from typing import Any, Dict, Optional

from .base import BaseTransformerModule
from ..utils import compute_metrics


class RelationClassificationTransformerModule(BaseTransformerModule):
    """
    TODO: Add docstring
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
                         cache_dir=cache_dir,
                         learning_rate=learning_rate, weight_decay=weight_decay,
                         adam_epsilon=adam_epsilon, warmup_steps=warmup_steps,
                         **kwargs)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=self.config,
            cache_dir=cache_dir
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def _loss(self, batch: Any) -> torch.Tensor:
        return self(**batch).loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.test_output_cache.append({
            "eval_loss": outputs.loss.item(),
            "true_labels": batch["labels"].tolist(),
            "pred_labels": outputs.logits.argmax(1).tolist(),
        })

    def on_test_epoch_end(self):
        eval_loss, true_labels, pred_labels = self._accumulate_test_results()

        # F1 macro and micro averages for classes different from the most common
        # one assuming the most common class is "0"
        outputs = compute_metrics(true_labels, pred_labels,
                                  [label for label in self.config.id2label if label != 0])
        outputs["eval_loss"] = np.mean(eval_loss)
        self.log_dict(outputs)
        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch.pop('labels', None)
        predictions = self(**batch).logits.argmax(1)

        return {
            "input_ids": batch.input_ids.tolist(),
            "labels": labels.tolist() if labels is not None else None,
            "predictions": predictions.tolist()
        }

"""
Sequence Tagging Transformer Module for Token Classification in Argumentation Mining.

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
import torch.nn as nn

from torchcrf import CRF
from transformers import AutoModel
from typing import Any, Dict, Optional

from .base import BaseTransformerModule
from ..utils import compute_metrics


class SequenceTaggingTransformerModule(BaseTransformerModule):
    """
    TODO: Add docstring
    """
    def __init__(self,
                 model_name_or_path: str,
                 label2id: Dict[str, int],
                 id2label: Dict[int, str],
                 config_name_or_path: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 masked_label: int = -100,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.0,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 **kwargs):
        super().__init__(model_name_or_path=model_name_or_path,
                         label2id=label2id, id2label=id2label,
                         config_name_or_path=config_name_or_path,
                         cache_dir=cache_dir, masked_label=masked_label,
                         learning_rate=learning_rate, weight_decay=weight_decay,
                         adam_epsilon=adam_epsilon, warmup_steps=warmup_steps,
                         **kwargs)

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            config=self.config,
            cache_dir=cache_dir
        )
        self.rnn = nn.GRU(self.config.hidden_size,
                          self.config.hidden_size,
                          batch_first=True,
                          bidirectional=True)
        self.linear = nn.Linear(2 * self.config.hidden_size,
                                self.config.num_labels)

        self.crf = CRF(self.config.num_labels,
                       batch_first=True)

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        rnn_out, _ = self.rnn(outputs[0])
        emissions = self.linear(rnn_out)
        path = torch.LongTensor(self.crf.decode(emissions))

        return path, emissions

    def _loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        labels = batch.pop('labels')
        path, emissions = self(**batch)
        mask = (labels != self.hparams.masked_label)

        return -self.crf(emissions, labels, mask=mask)

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        path, emissions = self(**batch)
        mask = (labels != self.hparams.masked_label)

        test_loss = -self.crf(emissions, labels, mask=mask)

        # Remove the masked labels
        non_mask_labels = torch.masked_select(labels, mask).tolist()
        non_masked_predictions = torch.masked_select(path, mask).tolist()
        assert len(non_mask_labels) == len(non_masked_predictions)

        self.test_output_cache.append({
            "test_loss": test_loss.item(),
            "true_labels": non_mask_labels,
            "pred_labels": non_masked_predictions,
        })

    def on_test_epoch_end(self):
        test_loss, true_labels, pred_labels = self._accumulate_test_results()

        # F1 macro and micro averages for classes different from the most common
        # one (this assumes the most common class is "0") and different from
        # masked label
        outputs = compute_metrics(true_labels, pred_labels,
                                  [label for label in self.config.id2label
                                   if label != 0 and label != self.hparams.masked_label])
        outputs["test_loss"] = np.mean(test_loss)
        self.log_dict(outputs)
        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch.pop('labels', None)
        path, emissions = self(**batch)

        return {
            "input_ids": batch.input_ids.tolist(),
            "labels": labels.tolist() if labels is not None else None,
            "predictions": path.tolist()
        }

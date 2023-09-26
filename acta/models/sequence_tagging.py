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

import torch
import torch.nn as nn

from torchcrf import CRF
from transformers import AutoModel
from typing import Any, Dict, Optional

from .base import BaseTransformerModule


class SequenceTaggingTransformerModule(BaseTransformerModule):
    """
    TODO: Add docstring
    """
    def __init__(self,
                 model_name_or_path: str,
                 id2label: Dict[int, str],
                 label2id: Dict[str, int],
                 config_name_or_path: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 masked_label: int = -100,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.0,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 **kwargs):
        super().__init__(model_name_or_path, id2label, label2id, config_name_or_path, cache_dir,
                         masked_label, learning_rate, weight_decay, adam_epsilon,
                         warmup_steps, **kwargs)

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
        mask = (labels != self.hparams.masked_label).to(torch.uint8)

        return -self.crf(emissions, labels, mask=mask)

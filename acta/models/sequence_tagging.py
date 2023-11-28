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
    Lightning Module for sequence tagging (i.e. classify each token in a
    sequence of tokens).

    It adds a Bidirectional Recurrent Neural Network (a GRU in this case) with a
    Linear Projection and uses Pytorch CRF for the loss for Sequence Tagging.

    This model was presented in the work of Mayer, Cabrio and Villata:
    "Transformer-based Argument Mining for Healthcare Applications" presented in
    ECAI 2020. For more information check: https://hal.science/hal-02879293/

    Parameters
    ----------
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
    masked_label_id: Optional[int]
        If given number (defaults to None), it masks the given label id in the
        CRF function.  This can fail if the given `masked_label_id` is present
        at the beggining of the sequence (e.g. if the special tokens for a
        transformer are not given the extension label but rather use the same
        'PAD' label that you are trying to mask).
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
                 masked_label_id: Optional[int] = None,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.0,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 **kwargs):
        super().__init__(model_name_or_path=model_name_or_path,
                         label2id=label2id, id2label=id2label,
                         config_name_or_path=config_name_or_path,
                         cache_dir=cache_dir, masked_label_id=masked_label_id,
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
        if self.hparams.masked_label_id is not None:
            mask = (labels != self.hparams.masked_label_id)
        else:
            mask = None

        return -self.crf(emissions, labels, mask=mask)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch.pop('labels', None)
        path, emissions = self(**batch)

        return {
            "input_ids": batch.input_ids.tolist(),
            "labels": labels.tolist() if labels is not None else None,
            "predictions": path.tolist()
        }

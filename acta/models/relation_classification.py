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

from transformers import AutoModelForSequenceClassification
from typing import Dict

from .base import BaseTransformerModule


class RelationClassificationTransformerModule(BaseTransformerModule):
    """
    TODO: Add docstring
    """
    def __init__(self,
                 model_name_or_path: str,
                 id2label: Dict[int, str],
                 label2id: Dict[str, int],
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.0,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 **kwargs):
        super().__init__(model_name_or_path, id2label, label2id,
                         learning_rate, weight_decay, adam_epsilon,
                         warmup_steps, **kwargs)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=self.config
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        return outputs.loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        self.log("val_loss", outputs.loss)

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("test_loss", outputs.loss)

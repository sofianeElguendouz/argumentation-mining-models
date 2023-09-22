"""
Pytorch Lightning Module for Relation Classfication.
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
                 warmup_steps: int = 0):
        super().__init__(model_name_or_path, id2label, label2id,
                         learning_rate, weight_decay, adam_epsilon,
                         warmup_steps)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=self.config
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        return outputs.loss

    # FIXME: Add validation, test and evaluation steps and logging

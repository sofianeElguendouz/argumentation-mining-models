"""
Module for abstract class which serves as the base model for combining Pytorch Lightning with
Hugging Face Transformers models.

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

import lightning.pytorch as pl

from abc import ABCMeta
from transformers import AdamW, AutoConfig, get_linear_schedule_with_warmup
from typing import Dict


class BaseTransformerModule(pl.LightningModule, metaclass=ABCMeta):
    """
    Abstract Base Class for a Transformer Module.

    Parameters
    ----------
    model_name_or_path: str
        The path or the name of a Transformer Model (from the HF repository).
    id2label: Dict[int, str]
        A mapping between indices and their corresponding labels (must be the
        reverse map of label2id).
    label2id: Dict[str, int]
        A mapping between labels and their corresponding indices (must be the
        reverse map of id2label).
    learning_rate: float
        The learning rate.
    weight_decay: float
        The weight decay for parameters that need it.
    adam_epsilon: float
        The Adam Epsilon constant.
    warmup_steps: int
        The number of warmup steps.
    """
    def __init__(self,
                 model_name_or_path: str,
                 id2label: Dict[int, str],
                 label2id: Dict[str, int],
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.0,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0):
        super().__init__()
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path,
                                                 num_labels=len(id2label),
                                                 id2label=id2label,
                                                 label2id=label2id)

    def configure_optimizers(self):
        """
        Method to prepare optimizer and scheduler (linear warmup and decay).
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

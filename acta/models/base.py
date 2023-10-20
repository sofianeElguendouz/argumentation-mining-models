"""
Module for the abstract class which serves as the base model for combining
Pytorch Lightning with Hugging Face Transformers models.

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
import torch

from abc import ABCMeta, abstractmethod
from transformers import AutoConfig, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import BatchEncoding
from typing import Dict, Optional


class BaseTransformerModule(pl.LightningModule, metaclass=ABCMeta):
    """
    Abstract Base Class for a Transformer Module.

    For more information check the LightningModule documentation and the
    tutorial on finetuning Hugging Face Transfomers with Pytorch Lightning:
    - https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    - https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/text-transformers.html

    Parameters
    ----------
    model_name_or_path: str
        The path or the name of a Transformer Model (from the HF repository).
    label2id: Dict[str, int]
        A mapping between labels and their corresponding indices (must be the
        reverse map of id2label).
    id2label: Dict[int, str]
        A mapping between indices and their corresponding labels (must be the
        reverse map of label2id).
    config_name_or_path: Optional[str]
        If given, uses this configuration instead of the one from the model.
    cache_dir: Optional[str]
        Directory to store the models.
    learning_rate: float
        The learning rate.
    weight_decay: float
        The weight decay for parameters that need it.
    adam_epsilon: float
        The Adam Epsilon constant.
    warmup_steps: int
        The number of warmup steps.
    **kwargs
        Extra keyword arguments dependant on the children classes.
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
        super().__init__()

        if len(label2id) != len(id2label) or\
                not all([k1 == v2 and k2 == v1 for (k1, v1), (k2, v2)
                         in zip(sorted(label2id.items(), key=lambda x: x[0]),
                                sorted(id2label.items(), key=lambda x: x[1]))]):
            raise ValueError("The parameters label2id and id2value are not the reverse "
                             "of each other")

        self.save_hyperparameters()

        config_name_or_path = config_name_or_path if config_name_or_path else model_name_or_path
        self.config = AutoConfig.from_pretrained(config_name_or_path,
                                                 num_labels=len(label2id),
                                                 label2id=label2id,
                                                 id2label=id2label,
                                                 cache_dir=cache_dir)

    @abstractmethod
    def _loss(self, batch: BatchEncoding) -> torch.Tensor:
        """
        Loss function calculation for a single batch of data.
        Returns the Tensor for the los value.

        Parameters
        ----------
        batch: BatchEncoding
            The same batch passed to `*_step`. It should have all the data
            needed to run the loss. In particular it is expected to have a
            `labels` key with the ground truth labels.

        Returns
        -------
        torch.Tensor
            The loss for the batch.
        """

    def training_step(self, batch, batch_idx):
        return self._loss(batch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._loss(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                      lr=self.hparams.learning_rate,
                                      eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

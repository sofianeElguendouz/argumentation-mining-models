import torch
import torch.nn as nn

from torchcrf import CRF
from transformers import AutoModel
from typing import Dict

from .base import BaseTransformerModule


class SequenceTaggingTransformerModule(BaseTransformerModule):
    """
    TODO: Add docstring
    """
    def __init__(self,
                 model_name_or_path: str,
                 id2label: Dict[int, str],
                 label2id: Dict[str, int],
                 masked_label: int = -100,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.0,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0):
        super().__init__(model_name_or_path, id2label, label2id,
                         learning_rate, weight_decay, adam_epsilon,
                         warmup_steps)

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            config=self.config
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

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        path, emissions = self(**batch)
        mask = (labels != self.hparams.masked_label).to(torch.uint8)

        return -self.crf(emissions, labels, mask=mask)

    # FIXME: Add validation, test and evaluation steps. Add logging.

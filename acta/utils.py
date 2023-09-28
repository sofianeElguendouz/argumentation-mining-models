"""
Utilities module for the ACTA library.

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

import sys

from lightning.pytorch.callbacks import TQDMProgressBar
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Optional, Union


def compute_metrics(true_labels: List[Union[int, str]],
                    pred_labels: List[Union[int, str]],
                    limited_labels: Optional[List[Union[int, str]]] = None) -> Dict[str, float]:
    """
    Helper function to compute F1-score and Accuracy metrics
    """
    outputs = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1_score_macro": f1_score(true_labels, pred_labels, average="macro", zero_division=0),
        "f1_score_micro": f1_score(true_labels, pred_labels, average="micro", zero_division=0)
    }

    if limited_labels is not None:
        # F1 macro and micro averages for specific classes
        outputs["f1_score_macro_limited"] = f1_score(
            true_labels, pred_labels, average="macro", zero_division=0, labels=limited_labels
        )
        outputs["f1_score_micro_limited"] = f1_score(
            true_labels, pred_labels, average="micro", zero_division=0, labels=limited_labels
        )
    return outputs


class TTYAwareProgressBar(TQDMProgressBar):
    """
    A Wrapper Callback around TQDMProgressBar that is TTY aware, thus using the
    TQDM bar or just printing to screen if not connected to a terminal.
    """
    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        bar.disable = None  # Deactivates for non TTY
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.disable = None  # Deactivates for non TTY
        return bar

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.disable = None  # Deactivates for non TTY
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = None  # Deactivates for non TTY
        return bar

    def _print_progress(self, stage: str, current_epoch: int, current_step: int,
                        total_steps: int, loss: float):
        """
        Prints progress to the standard output.
        """
        percent = (current_step / total_steps) * 100
        sys.stdout.write(
            f"{stage.capitalize()} Epoch {current_epoch} - "
            f"Step {current_step} of {total_steps} ({percent:.2f}%) - "
            f"Loss: {loss:.4f}\n"
        )
        sys.stdout.flush()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        current = batch_idx + 1
        if not sys.stdout.isatty() and \
                (current % self.refresh_rate == 0 or current == self.train_progress_bar.total):
            self._print_progress("Training", trainer.current_epoch, current,
                                 self.train_progress_bar.total, outputs['loss'])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        current = batch_idx + 1
        if not sys.stdout.isatty() and \
                (current % self.refresh_rate == 0 or current == self.test_progress_bar.total):
            self._print_progress("Test", trainer.current_epoch, current,
                                 self.test_progress_bar.total, outputs['test_loss'])

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                dataloader_idx=0):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx,
                                        dataloader_idx)
        current = batch_idx + 1
        if not sys.stdout.isatty() and \
                (current % self.refresh_rate == 0 or current == self.val_progress_bar.total):
            self._print_progress("Validation", trainer.current_epoch, current,
                                 self.val_progress_bar.total, outputs['val_loss'])

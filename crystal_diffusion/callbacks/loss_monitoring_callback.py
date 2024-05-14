from typing import Any, AnyStr, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import Callback

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.loggers.logger_loader import log_figure

plt.style.use(PLOT_STYLE_PATH)


def instantiate_loss_monitoring_callback(callback_params: Dict[AnyStr, Any],
                                         output_directory: str, verbose: bool) -> Dict[str, Callback]:
    """Instantiate the Loss monitoring callback."""
    loss_monitoring_callback = LossMonitoringCallback()
    return dict(loss_monitoring_callback=loss_monitoring_callback)


class LossMonitoringCallback(Callback):
    """Callback class to monitor the loss vs. time (or sigma) relationship."""

    def __init__(self):
        """Init method."""
        self.all_sigmas = []
        self.all_squared_errors = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Action to perform at the end of a validation batch."""
        batch_sigmas = outputs['sigmas'][:, :, 0]  # the sigmas are the same for all atoms and space directions
        self.all_sigmas.append(batch_sigmas.flatten())

        # Compute the square errors per atoms
        batched_squared_errors = ((outputs['predicted_normalized_scores']
                                   - outputs['target_normalized_conditional_scores'])**2).sum(dim=-1)
        self.all_squared_errors.append(batched_squared_errors.flatten())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Action to perform at the end of a training epoch."""
        # free up the memory

        sigmas = torch.cat(self.all_sigmas).detach().cpu().numpy()
        squared_errors = torch.cat(self.all_squared_errors).detach().cpu().numpy()

        fig = self._plot_loss_scatter(sigmas, squared_errors)

        for pl_logger in trainer.loggers:
            log_figure(figure=fig,
                       global_step=trainer.global_step,
                       pl_logger=pl_logger,
                       dataset="validation",
                       name="squared_error")

        self.all_sigmas.clear()
        self.all_squared_errors.clear()

    @staticmethod
    def _plot_loss_scatter(sigmas: np.ndarray, squared_errors: np.array) -> plt.figure:
        """Generate a scatter plot of the squared errors vs. the values of noise."""
        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        fig.suptitle('Loss vs. Sigma')

        ax1 = fig.add_subplot(111)

        ax1.plot(sigmas, squared_errors, 'bo')
        ax1.set_xlabel('$\\sigma$$')
        ax1.set_ylabel('Squared Error')
        fig.tight_layout()
        return fig

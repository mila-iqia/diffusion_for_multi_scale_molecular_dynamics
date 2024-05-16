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
    number_of_bins = callback_params['number_of_bins']
    sample_every_n_epochs = callback_params['sample_every_n_epochs']
    spatial_dimension = callback_params.get('spatial_dimension', 3)
    loss_monitoring_callback = LossMonitoringCallback(number_of_bins=number_of_bins,
                                                      sample_every_n_epochs=sample_every_n_epochs,
                                                      spatial_dimension=spatial_dimension)
    return dict(loss_monitoring_callback=loss_monitoring_callback)


class LossMonitoringCallback(Callback):
    """Callback class to monitor the loss vs. time (or sigma) relationship."""

    def __init__(self, number_of_bins: int, sample_every_n_epochs: int, spatial_dimension: int = 3):
        """Init method."""
        self.number_of_bins = number_of_bins
        self.sample_every_n_epochs = sample_every_n_epochs
        self.spatial_dimension = spatial_dimension
        self.all_sigmas = []
        self.all_squared_errors = []

    def _compute_results_at_this_epoch(self, current_epoch: int) -> bool:
        """Check if results should be computed at this epoch."""
        # Do not produce results at epoch 0; it would be meaningless.
        if current_epoch % self.sample_every_n_epochs == 0 and current_epoch > 0:
            return True
        else:
            return False

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Action to perform at the end of a validation batch."""
        if not self._compute_results_at_this_epoch(trainer.current_epoch):
            return
        batch_sigmas = outputs['sigmas'][:, :, 0]  # the sigmas are the same for all atoms and space directions
        self.all_sigmas.append(batch_sigmas.flatten())

        # Compute the square errors per atoms
        batched_squared_errors = ((outputs['predicted_normalized_scores']
                                   - outputs['target_normalized_conditional_scores'])**2).sum(dim=-1)
        self.all_squared_errors.append(batched_squared_errors.flatten())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Action to perform at the end of a training epoch."""
        if not self._compute_results_at_this_epoch(trainer.current_epoch):
            return

        sigmas = torch.cat(self.all_sigmas).detach().cpu().numpy()
        squared_errors = torch.cat(self.all_squared_errors).detach().cpu().numpy()

        fig = self._plot_loss_scatter(sigmas, squared_errors, trainer.current_epoch)

        for pl_logger in trainer.loggers:
            log_figure(figure=fig,
                       global_step=trainer.global_step,
                       pl_logger=pl_logger,
                       dataset="validation",
                       name="squared_error")

        self.all_sigmas.clear()
        self.all_squared_errors.clear()

    def _plot_loss_scatter(self, sigmas: np.ndarray, squared_errors: np.array, epoch: int) -> plt.figure:
        """Generate a scatter plot of the squared errors vs. the values of noise."""
        loss = np.mean(squared_errors) / self.spatial_dimension
        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        fig.suptitle(f'Loss vs. Sigma\nEpoch {epoch}, Loss = {loss:5.3e}')

        bins = np.linspace(0., np.max(sigmas), self.number_of_bins)
        bin_indices = np.digitize(sigmas, bins)

        list_mean_sigmas = []
        list_mean_squared_errors = []
        list_min_squared_errors = []
        list_max_squared_errors = []
        for index in np.unique(bin_indices):
            mask = bin_indices == index
            bin_sigmas = sigmas[mask]
            bin_squared_errors = squared_errors[mask]

            list_mean_sigmas.append(np.mean(bin_sigmas))
            list_mean_squared_errors.append(np.mean(bin_squared_errors))
            list_min_squared_errors.append(np.quantile(bin_squared_errors, q=0.05))
            list_max_squared_errors.append(np.quantile(bin_squared_errors, q=0.95))

        list_mean_sigmas = np.array(list_mean_sigmas)
        list_mean_squared_errors = np.array(list_mean_squared_errors)
        list_min_squared_errors = np.array(list_min_squared_errors)
        list_max_squared_errors = np.array(list_max_squared_errors)

        ax1 = fig.add_subplot(111)

        ax1.semilogy(list_mean_sigmas, list_mean_squared_errors, 'g-o', label='Binned Mean')
        ax1.fill_between(list_mean_sigmas,
                         list_min_squared_errors,
                         list_max_squared_errors,
                         color='g', alpha=0.25, label="5% to 95% Quantile")

        ax1.set_xlim([-0.01, np.max(sigmas) + 0.01])
        ax1.legend(loc='best')
        ax1.set_xlabel('$\\sigma$')
        ax1.set_ylabel('Squared Error')
        fig.tight_layout()
        return fig

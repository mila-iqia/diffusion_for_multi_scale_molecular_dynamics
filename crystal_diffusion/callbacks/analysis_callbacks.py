"""Analysis Callbacks.

The callbacks in this module are not designed for 'production'. They are to be used in
ad hoc / debugging / analysis experiments.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from crystal_diffusion.analysis import PLOT_STYLE_PATH
from crystal_diffusion.analysis.analytic_score.utils import \
    get_relative_harmonic_energy
from crystal_diffusion.callbacks.sampling_callback import (
    DiffusionSamplingCallback, SamplingParameters)
from crystal_diffusion.samplers.variance_sampler import NoiseParameters

logger = logging.getLogger(__name__)

plt.style.use(PLOT_STYLE_PATH)


class HarmonicEnergyDiffusionSamplingCallback(DiffusionSamplingCallback):
    """Callback class to periodically generate samples and log their energies."""

    def __init__(self, noise_parameters: NoiseParameters,
                 sampling_parameters: SamplingParameters,
                 equilibrium_relative_coordinates: torch.Tensor,
                 inverse_covariance: torch.Tensor,
                 output_directory: str):
        """Init method."""
        super().__init__(noise_parameters, sampling_parameters, output_directory)

        self.equilibrium_relative_coordinates = equilibrium_relative_coordinates
        self.inverse_covariance = inverse_covariance

    def _compute_oracle_energies(self, batch_relative_coordinates: torch.Tensor) -> np.ndarray:
        """Compute energies from samples."""
        logger.info("Compute harmonic energy from Oracle")

        spring_constant = self.inverse_covariance[0, 0, 0, 0]

        energies = get_relative_harmonic_energy(batch_relative_coordinates,
                                                self.equilibrium_relative_coordinates,
                                                spring_constant)

        # energies = get_samples_harmonic_energy(self.equilibrium_relative_coordinates,
        #                                        self.inverse_covariance,
        #                                        batch_relative_coordinates)
        return energies.cpu().detach().numpy()

    @staticmethod
    def _plot_energy_histogram(sample_energies: np.ndarray, validation_dataset_energies: np.array,
                               epoch: int) -> plt.figure:
        fig = DiffusionSamplingCallback._plot_energy_histogram(sample_energies, validation_dataset_energies, epoch)

        fig.suptitle(f'Sampling Unitless Harmonic Potential Energy Distributions\nEpoch {epoch}')
        ax1 = fig.axes[0]
        ax1.set_xlabel('Unitless Harmonic Energy')
        return fig

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AnyStr, Dict, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import Callback, LightningModule, Trainer

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE
from crystal_diffusion.loggers.logger_loader import log_figure
from crystal_diffusion.oracle.lammps import get_energy_and_forces_from_lammps
from crystal_diffusion.samplers.predictor_corrector_position_sampler import \
    AnnealedLangevinDynamicsSampler
from crystal_diffusion.samplers.variance_sampler import NoiseParameters

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SamplingParameters:
    """Hyper-parameters for diffusion sampling."""
    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live.
    number_of_corrector_steps: int = 1
    number_of_atoms: int  # the number of atoms that must be generated in a sampled configuration.
    number_of_samples: int
    sample_every_n_epochs: int = 1  # Sampling is expensive; control frequency
    cell_dimensions: List[float]  # unit cell dimensions; the unit cell is assumed to be a orthogonal.


def instantiate_diffusion_sampling_callback(callback_params: Dict[AnyStr, Any],
                                            output_directory: str,
                                            verbose: bool) -> Dict[str, Callback]:
    """Instantiate the Diffusion Sampling callback."""
    noise_parameters = NoiseParameters(**callback_params['noise'])
    sampling_parameters = SamplingParameters(**callback_params['sampling'])

    sample_output_directory = os.path.join(output_directory, 'energy_samples')
    Path(sample_output_directory).mkdir(parents=True, exist_ok=True)

    diffusion_sampling_callback = DiffusionSamplingCallback(noise_parameters=noise_parameters,
                                                            sampling_parameters=sampling_parameters,
                                                            output_directory=sample_output_directory)

    return dict(diffusion_sampling=diffusion_sampling_callback)


class DiffusionSamplingCallback(Callback):
    """Callback class to periodically generate samples and log their energies."""

    def __init__(self, noise_parameters: NoiseParameters,
                 sampling_parameters: SamplingParameters,
                 output_directory: str):
        """Init method."""
        self.noise_parameters = noise_parameters
        self.sampling_parameters = sampling_parameters
        self.output_directory = output_directory

    def _draw_sample_of_relative_positions(self, pl_model: LightningModule) -> np.ndarray:
        """Draw a sample from the generative model."""
        logger.info("Creating sampler")
        sigma_normalized_score_network = pl_model.sigma_normalized_score_network

        sampler_parameters = dict(noise_parameters=self.noise_parameters,
                                  number_of_corrector_steps=self.sampling_parameters.number_of_corrector_steps,
                                  number_of_atoms=self.sampling_parameters.number_of_atoms,
                                  spatial_dimension=self.sampling_parameters.spatial_dimension)

        pc_sampler = AnnealedLangevinDynamicsSampler(sigma_normalized_score_network=sigma_normalized_score_network,
                                                     **sampler_parameters)
        logger.info("Draw samples")
        samples = pc_sampler.sample(self.sampling_parameters.number_of_samples)

        batch_relative_positions = samples.cpu().numpy()
        return batch_relative_positions

    @staticmethod
    def _plot_energy_histogram(sample_energies: np.ndarray) -> plt.figure:
        """Generate a plot of the energy samples."""
        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)

        fig.suptitle('Sampling Energy Distributions')

        ax1 = fig.add_subplot(111)

        ax1.hist(sample_energies, density=True, bins=50, histtype="stepfilled", alpha=0.25, color='green')
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Density')
        fig.tight_layout()
        return fig

    def _compute_lammps_energies(self, batch_relative_positions: np.ndarray) -> np.ndarray:
        """Compute energies from samples."""
        box = np.diag(self.sampling_parameters.cell_dimensions)
        batch_positions = np.dot(batch_relative_positions, box)
        atom_types = np.ones(self.sampling_parameters.number_of_atoms, dtype=int)

        list_energy = []

        logger.info("Compute energy from Oracle")

        with tempfile.TemporaryDirectory() as tmp_work_dir:
            for idx, positions in enumerate(batch_positions):
                energy, forces = get_energy_and_forces_from_lammps(positions,
                                                                   box,
                                                                   atom_types,
                                                                   tmp_work_dir=tmp_work_dir)
                list_energy.append(energy)

        return np.array(list_energy)

    def on_validation_epoch_end(self, trainer: Trainer, pl_model: LightningModule) -> None:
        """On validation epoch end."""
        if trainer.current_epoch % self.sampling_parameters.sample_every_n_epochs != 0:
            return

        batch_relative_positions = self._draw_sample_of_relative_positions(pl_model)
        sample_energies = self._compute_lammps_energies(batch_relative_positions)

        output_path = os.path.join(self.output_directory, f"energies_sample_epoch={trainer.current_epoch}.pt")
        torch.save(torch.from_numpy(sample_energies), output_path)

        fig = self._plot_energy_histogram(sample_energies)

        for pl_logger in trainer.loggers:
            log_figure(figure=fig, global_step=trainer.global_step, pl_logger=pl_logger)

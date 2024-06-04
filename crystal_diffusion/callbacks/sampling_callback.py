import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AnyStr, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as ss
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import Callback, LightningModule, Trainer

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.loggers.logger_loader import log_figure
from crystal_diffusion.oracle.lammps import get_energy_and_forces_from_lammps
from crystal_diffusion.samplers.predictor_corrector_position_sampler import \
    AnnealedLangevinDynamicsSampler
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.basis_transformations import \
    get_positions_from_coordinates

logger = logging.getLogger(__name__)


plt.style.use(PLOT_STYLE_PATH)


@dataclass(kw_only=True)
class SamplingParameters:
    """Hyper-parameters for diffusion sampling."""
    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live.
    number_of_corrector_steps: int = 1
    number_of_atoms: int  # the number of atoms that must be generated in a sampled configuration.
    number_of_samples: int
    sample_batchsize: Optional[int] = None  # iterate up to number_of_samples with batches of this size
    # if None, use number_of_samples as batchsize
    sample_every_n_epochs: int = 1  # Sampling is expensive; control frequency
    cell_dimensions: List[float]  # unit cell dimensions; the unit cell is assumed to be an orthogonal box.
    record_samples: bool = False  # should the predictor and corrector steps be recorded to a file


def instantiate_diffusion_sampling_callback(callback_params: Dict[AnyStr, Any],
                                            output_directory: str,
                                            verbose: bool) -> Dict[str, Callback]:
    """Instantiate the Diffusion Sampling callback."""
    noise_parameters = NoiseParameters(**callback_params['noise'])

    sampling_parameters = SamplingParameters(**callback_params['sampling'])

    diffusion_sampling_callback = DiffusionSamplingCallback(noise_parameters=noise_parameters,
                                                            sampling_parameters=sampling_parameters,
                                                            output_directory=output_directory)

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

        self.energy_sample_output_directory = os.path.join(output_directory, 'energy_samples')
        Path(self.energy_sample_output_directory).mkdir(parents=True, exist_ok=True)

        if self.sampling_parameters.record_samples:
            self.position_sample_output_directory = os.path.join(output_directory, 'diffusion_position_samples')
            Path(self.position_sample_output_directory).mkdir(parents=True, exist_ok=True)

        self._initialize_validation_energies_array()

    @staticmethod
    def _get_orthogonal_unit_cell(batch_size: int, cell_dimensions: List[float]) -> torch.Tensor:
        """Get orthogonal unit cell.

        Args:
            batch_size: number of required repetitions of the unit cell.
            cell_dimensions : list of dimensions that correspond to the sides of the unit cell.

        Returns:
            unit_cell: a diagonal matrix with the dimensions along the diagonal.
        """
        unit_cell = torch.diag(torch.Tensor(cell_dimensions)).unsqueeze(0).repeat(batch_size, 1, 1)
        return unit_cell

    @staticmethod
    def compute_kolmogorov_smirnov_distance_and_pvalue(sampling_energies: np.ndarray,
                                                       reference_energies: np.ndarray) -> Tuple[float, float]:
        """Compute Kolmogorov Smirnov Distance.

        Compute the two sample Kolmogorov–Smirnov test in order to gauge whether the
        sample_energies sample was drawn from the same distribution as the reference_energies.

        Args:
            sampling_energies : a sample of energies drawn from the diffusion model.
            reference_energies :a sample of energies drawn from the reference distribution.

        Returns:
            ks_distance, p_value: the Kolmogorov-Smirnov test statistic (a "distance")
                and the statistical test's p-value.
        """
        test_result = ss.ks_2samp(sampling_energies, reference_energies, alternative='two-sided', method='auto')

        # The "test statistic" of the two-sided KS test is the largest vertical distance between
        # the empirical CDFs of the two samples. The larger this is, the less likely the two
        # samples were drawn from the same underlying distribution, hence the idea of 'distance'.
        ks_distance = test_result.statistic

        # The null hypothesis of the KS test is that both samples are drawn from the same distribution.
        # Thus, a small p-value (which leads to the rejection of the null hypothesis) indicates that
        # the samples probably come from different distributions (ie, our samples are bad!).
        p_value = test_result.pvalue
        return ks_distance, p_value

    def _compute_results_at_this_epoch(self, current_epoch: int) -> bool:
        """Check if results should be computed at this epoch."""
        # Do not produce results at epoch 0; it would be meaningless.
        if current_epoch % self.sampling_parameters.sample_every_n_epochs == 0 and current_epoch > 0:
            return True
        else:
            return False

    def _initialize_validation_energies_array(self):
        """Initialize the validation energies array to an empty array."""
        # The validation energies will be extracted at epochs where it is needed. Although this
        # data does not change, we will avoid having this in memory at all times.
        self.validation_energies = np.array([])

    def _create_sampler(self, pl_model: LightningModule) -> Tuple[AnnealedLangevinDynamicsSampler, torch.Tensor]:
        """Draw a sample from the generative model."""
        logger.info("Creating sampler")
        sigma_normalized_score_network = pl_model.sigma_normalized_score_network

        sampler_parameters = dict(noise_parameters=self.noise_parameters,
                                  number_of_corrector_steps=self.sampling_parameters.number_of_corrector_steps,
                                  number_of_atoms=self.sampling_parameters.number_of_atoms,
                                  spatial_dimension=self.sampling_parameters.spatial_dimension,
                                  record_samples=self.sampling_parameters.record_samples,
                                  positions_require_grad=pl_model.grads_are_needed_in_inference)

        pc_sampler = AnnealedLangevinDynamicsSampler(sigma_normalized_score_network=sigma_normalized_score_network,
                                                     **sampler_parameters)
        # TODO we will have to sample unit cell dimensions at some points instead of working with fixed size
        unit_cell = (self._get_orthogonal_unit_cell(batch_size=self.sampling_parameters.number_of_samples,
                                                    cell_dimensions=self.sampling_parameters.cell_dimensions)
                     .to(pl_model.device))

        return pc_sampler, unit_cell

    @staticmethod
    def _plot_energy_histogram(sample_energies: np.ndarray, validation_dataset_energies: np.array,
                               epoch: int) -> plt.figure:
        """Generate a plot of the energy samples."""
        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)

        minimum_energy = validation_dataset_energies.min()
        maximum_energy = validation_dataset_energies.max()
        energy_range = maximum_energy - minimum_energy

        emin = minimum_energy - 0.2 * energy_range
        emax = maximum_energy + 0.2 * energy_range
        bins = np.linspace(emin, emax, 101)

        number_of_samples_in_range = np.logical_and(sample_energies >= emin, sample_energies <= emax).sum()

        fig.suptitle(f'Sampling Energy Distributions\nEpoch {epoch}')

        common_params = dict(density=True, bins=bins, histtype="stepfilled", alpha=0.25)

        ax1 = fig.add_subplot(111)

        ax1.hist(sample_energies, **common_params,
                 label=f'Samples \n(total count = {len(sample_energies)}, in range = {number_of_samples_in_range})',
                 color='red')
        ax1.hist(validation_dataset_energies, **common_params,
                 label=f'Validation Data \n(count = {len(validation_dataset_energies)})', color='green')

        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Density')
        ax1.legend(loc='upper right', fancybox=True, shadow=True, ncol=1, fontsize=6)
        fig.tight_layout()
        return fig

    def _compute_lammps_energies(self, batch_relative_coordinates: torch.Tensor) -> np.ndarray:
        """Compute energies from samples."""
        batch_size = batch_relative_coordinates.shape[0]
        cell_dimensions = self.sampling_parameters.cell_dimensions
        basis_vectors = self._get_orthogonal_unit_cell(batch_size, cell_dimensions)
        batch_cartesian_positions = get_positions_from_coordinates(batch_relative_coordinates, basis_vectors)

        atom_types = np.ones(self.sampling_parameters.number_of_atoms, dtype=int)

        list_energy = []

        logger.info("Compute energy from Oracle")

        with tempfile.TemporaryDirectory() as tmp_work_dir:
            for positions, box in zip(batch_cartesian_positions.numpy(), basis_vectors.numpy()):
                energy, forces = get_energy_and_forces_from_lammps(positions,
                                                                   box,
                                                                   atom_types,
                                                                   tmp_work_dir=tmp_work_dir)
                list_energy.append(energy)

        return np.array(list_energy)

    def sample_and_evaluate_energy(self, pl_model: LightningModule, current_epoch: int = 0) -> np.ndarray:
        """Create samples and estimate their energy with an oracle (LAMMPS)

        Args:
            pl_model: pytorch-lightning model
            current_epoch (optional): current epoch to save files. Defaults to 0.

        Returns:
            array with energy of each sample from LAMMPS
        """
        pc_sampler, unit_cell = self._create_sampler(pl_model)

        logger.info("Draw samples")

        if self.sampling_parameters.sample_batchsize is None:
            self.sampling_parameters.sample_batchsize = self.sampling_parameters.number_of_samples

        sample_energies = []

        for n in range(0, self.sampling_parameters.number_of_samples, self.sampling_parameters.sample_batchsize):
            unit_cell_ = unit_cell[n:min(n + self.sampling_parameters.sample_batchsize,
                                         self.sampling_parameters.number_of_samples)]
            samples = pc_sampler.sample(min(self.sampling_parameters.number_of_samples - n,
                                            self.sampling_parameters.sample_batchsize),
                                        device=pl_model.device,
                                        unit_cell=unit_cell_)
            if self.sampling_parameters.record_samples:
                sample_output_path = os.path.join(self.position_sample_output_directory,
                                                  f"diffusion_position_sample_epoch={current_epoch}"
                                                  + f"_steps={n}.pt")
                # write trajectories to disk and reset to save memory
                pc_sampler.sample_trajectory_recorder.write_to_pickle(sample_output_path)
                pc_sampler.sample_trajectory_recorder.reset()
            batch_relative_coordinates = samples.detach().cpu()
            sample_energies += [self._compute_lammps_energies(batch_relative_coordinates)]

        sample_energies = np.concatenate(sample_energies)

        return sample_energies

    def on_validation_batch_start(self, trainer: Trainer,
                                  pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        """On validation batch start, accumulate the validation dataset energies for further processing."""
        if not self._compute_results_at_this_epoch(trainer.current_epoch):
            return
        self.validation_energies = np.append(self.validation_energies, batch['potential_energy'].cpu().numpy())

    def on_validation_epoch_end(self, trainer: Trainer, pl_model: LightningModule) -> None:
        """On validation epoch end."""
        if not self._compute_results_at_this_epoch(trainer.current_epoch):
            return

        # generate samples and evaluate their energy with an oracle
        sample_energies = self.sample_and_evaluate_energy(pl_model, trainer)

        energy_output_path = os.path.join(self.energy_sample_output_directory,
                                          f"energies_sample_epoch={trainer.current_epoch}.pt")
        torch.save(torch.from_numpy(sample_energies), energy_output_path)

        fig = self._plot_energy_histogram(sample_energies, self.validation_energies, trainer.current_epoch)
        ks_distance, p_value = self.compute_kolmogorov_smirnov_distance_and_pvalue(sample_energies,
                                                                                   self.validation_energies)

        pl_model.log("validation_epoch_ks_distance", ks_distance, on_step=False, on_epoch=True)
        pl_model.log("validation_epoch_ks_p_value", p_value, on_step=False, on_epoch=True)

        for pl_logger in trainer.loggers:
            log_figure(figure=fig, global_step=trainer.global_step, pl_logger=pl_logger)

        self._initialize_validation_energies_array()

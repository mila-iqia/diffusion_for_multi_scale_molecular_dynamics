import logging
import tempfile

import einops
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics import ANALYSIS_RESULTS_DIR
from diffusion_for_multi_scale_molecular_dynamics.analysis import \
    PLOT_STYLE_PATH
from diffusion_for_multi_scale_molecular_dynamics.callbacks.loss_monitoring_callback import \
    LossMonitoringCallback
from diffusion_for_multi_scale_molecular_dynamics.callbacks.sampling_visualization_callback import \
    PredictorCorrectorDiffusionSamplingCallback
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_position_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.loss import (
    MSELossParameters, create_loss_calculator)
from diffusion_for_multi_scale_molecular_dynamics.models.optimizer import \
    OptimizerParameters
from diffusion_for_multi_scale_molecular_dynamics.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.scheduler import \
    CosineAnnealingLRSchedulerParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters,
    TargetScoreBasedAnalyticalScoreNetwork)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    CARTESIAN_FORCES, RELATIVE_COORDINATES)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.variance_sampler import (
    ExplodingVarianceSampler, NoiseParameters)
from diffusion_for_multi_scale_molecular_dynamics.noisy_targets.noisy_relative_coordinates_sampler import \
    NoisyRelativeCoordinatesSampler
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps import \
    get_energy_and_forces_from_lammps
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from experiments.analysis.analytic_score.utils import (get_exact_samples,
                                                       get_silicon_supercell)

logger = logging.getLogger(__name__)


class AnalyticalScorePositionDiffusionLightningModel(PositionDiffusionLightningModel):
    """Analytical Score Position Diffusion Lightning Model.

    Overload the base class so that we can properly feed in an analytical score network.
    This should not be in the main code as the analytical score is not a real model.
    """

    def __init__(self, hyper_params: PositionDiffusionParameters):
        """Init method.

        This initializes the class.
        """
        pl.LightningModule.__init__(self)

        self.hyper_params = hyper_params
        self.save_hyperparameters(logger=False)

        self.use_permutation_invariance = (
            hyper_params.score_network_parameters.use_permutation_invariance
        )

        if self.use_permutation_invariance:
            self.sigma_normalized_score_network = AnalyticalScoreNetwork(
                hyper_params.score_network_parameters
            )
        else:
            self.sigma_normalized_score_network = (
                TargetScoreBasedAnalyticalScoreNetwork(
                    hyper_params.score_network_parameters
                )
            )

        self.loss_calculator = create_loss_calculator(hyper_params.loss_parameters)
        self.noisy_relative_coordinates_sampler = NoisyRelativeCoordinatesSampler()
        self.variance_sampler = ExplodingVarianceSampler(hyper_params.noise_parameters)

    def on_validation_start(self) -> None:
        """On validation start."""
        if self.use_permutation_invariance:
            # In this case, the analytical score is computed by explicitly calling autograd.
            torch.set_grad_enabled(True)


plt.style.use(PLOT_STYLE_PATH)

device = torch.device("cpu")

dataset_size = 1024
batch_size = 1024

spatial_dimension = 3
kmax = 8
supercell_factor = 1
kmax_target_score = 4

acell = 5.43

cell_dimensions = 3 * [supercell_factor * acell]

use_permutation_invariance = False
use_equilibrium = False
if use_equilibrium:
    model_variance_parameter = 0.0
else:
    # samples will be created
    sigma_d = 0.0025 / np.sqrt(supercell_factor)
    variance_parameter = sigma_d**2
    model_variance_parameter = 1.0 * variance_parameter


noise_parameters = NoiseParameters(
    total_time_steps=1000, sigma_min=0.0001, sigma_max=0.5
)

# We will not optimize, so  this doesn't matter
dummy_optimizer_parameters = OptimizerParameters(
    name="adam", learning_rate=0.001, weight_decay=0.0
)
dummy_scheduler_parameters = CosineAnnealingLRSchedulerParameters(T_max=10)

loss_monitoring_callback = LossMonitoringCallback(
    number_of_bins=100, sample_every_n_epochs=1, spatial_dimension=spatial_dimension
)

experiment_name = (
    f"cell_{supercell_factor}x{supercell_factor}x{supercell_factor}"
    f"_permutation_{use_permutation_invariance}"
    f"_sigma_d={np.sqrt(model_variance_parameter):5.4f}_super_noise"
)

output_dir = ANALYSIS_RESULTS_DIR / "PERFECT_SCORE_LOSS"

csv_logger = CSVLogger(save_dir=str(output_dir), name=experiment_name)

if __name__ == "__main__":

    box = torch.diag(torch.tensor(cell_dimensions))

    equilibrium_relative_coordinates = torch.from_numpy(
        get_silicon_supercell(supercell_factor=supercell_factor)
    ).to(torch.float32)

    number_of_atoms, _ = equilibrium_relative_coordinates.shape
    nd = number_of_atoms * spatial_dimension

    atom_types = np.array(number_of_atoms * [1])

    sampling_parameters = PredictorCorrectorSamplingParameters(
        number_of_atoms=number_of_atoms,
        number_of_corrector_steps=10,
        number_of_samples=batch_size,
        sample_batchsize=batch_size,
        sample_every_n_epochs=1,
        first_sampling_epoch=0,
        cell_dimensions=cell_dimensions,
        record_samples=False,
    )

    diffusion_sampling_callback = PredictorCorrectorDiffusionSamplingCallback(
        noise_parameters=noise_parameters,
        sampling_parameters=sampling_parameters,
        output_directory=output_dir / experiment_name,
    )

    if use_equilibrium:
        exact_samples = einops.repeat(
            equilibrium_relative_coordinates, "n d -> b n d", b=dataset_size
        )
    else:
        inverse_covariance = torch.diag(torch.ones(nd)) / variance_parameter
        inverse_covariance = inverse_covariance.reshape(
            number_of_atoms, spatial_dimension, number_of_atoms, spatial_dimension
        )

        # Create a dataloader
        exact_samples = get_exact_samples(
            equilibrium_relative_coordinates, inverse_covariance, dataset_size
        )
        exact_samples = map_relative_coordinates_to_unit_cell(exact_samples)

    list_oracle_energies = []
    list_positions = (exact_samples @ box).cpu().numpy()
    with tempfile.TemporaryDirectory() as tmp_work_dir:
        for positions in tqdm(list_positions, "LAMMPS energies"):
            energy, forces = get_energy_and_forces_from_lammps(
                positions, box.numpy(), atom_types, tmp_work_dir=tmp_work_dir
            )
            list_oracle_energies.append(energy)

    dataset = [
        {
            RELATIVE_COORDINATES: x,
            CARTESIAN_FORCES: torch.zeros_like(x),
            "potential_energy": energy,
            "box": torch.tensor(cell_dimensions),
        }
        for x, energy in zip(exact_samples, list_oracle_energies)
    ]

    dataloader = DataLoader(dataset, batch_size=batch_size)

    score_network_parameters = AnalyticalScoreNetworkParameters(
        number_of_atoms=number_of_atoms,
        spatial_dimension=spatial_dimension,
        use_permutation_invariance=use_permutation_invariance,
        kmax=kmax,
        equilibrium_relative_coordinates=equilibrium_relative_coordinates,
        variance_parameter=model_variance_parameter,
    )

    diffusion_params = PositionDiffusionParameters(
        score_network_parameters=score_network_parameters,
        loss_parameters=MSELossParameters(),
        optimizer_parameters=dummy_optimizer_parameters,
        scheduler_parameters=dummy_scheduler_parameters,
        noise_parameters=noise_parameters,
        kmax_target_score=kmax_target_score,
    )

    model = AnalyticalScorePositionDiffusionLightningModel(diffusion_params)

    trainer = pl.Trainer(
        callbacks=[loss_monitoring_callback, diffusion_sampling_callback],
        max_epochs=1,
        log_every_n_steps=1,
        fast_dev_run=False,
        logger=csv_logger,
        inference_mode=not use_equilibrium,  # Do this or else the gradients don't work!
    )

    # Run a validation epoch
    trainer.validate(model, dataloaders=[dataloader])

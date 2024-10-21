import logging

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.analysis.analytic_score.exploring_langevin_generator.generate_sample_energies import \
    EnergyCalculator
from crystal_diffusion.analysis.analytic_score.utils import \
    get_silicon_supercell
from crystal_diffusion.generators.langevin_generator import LangevinGenerator
from crystal_diffusion.generators.ode_position_generator import (
    ExplodingVarianceODEPositionGenerator, ODESamplingParameters)
from crystal_diffusion.generators.predictor_corrector_position_generator import \
    PredictorCorrectorSamplingParameters
from crystal_diffusion.generators.sde_position_generator import (
    ExplodingVarianceSDEPositionGenerator, SDESamplingParameters)
from crystal_diffusion.models.position_diffusion_lightning_model import \
    PositionDiffusionLightningModel
from crystal_diffusion.models.score_networks import ScoreNetwork
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.logging_utils import setup_analysis_logger

plt.style.use(PLOT_STYLE_PATH)

logger = logging.getLogger(__name__)
setup_analysis_logger()


class ForcedStartingPointLangevinGenerator(LangevinGenerator):
    """Langevin Generator with Forced Starting point."""
    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: PredictorCorrectorSamplingParameters,
        sigma_normalized_score_network: ScoreNetwork,
        starting_relative_coordinates: torch.Tensor,
    ):
        """Init method."""
        super().__init__(
            noise_parameters, sampling_parameters, sigma_normalized_score_network
        )

        self._starting_relative_coordinates = starting_relative_coordinates

    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        relative_coordinates = einops.repeat(
            self._starting_relative_coordinates,
            "natoms space -> batch_size natoms space",
            batch_size=number_of_samples,
        )
        return relative_coordinates


class ForcedStartingPointODEPositionGenerator(ExplodingVarianceODEPositionGenerator):
    """Forced starting point ODE position generator."""
    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: ODESamplingParameters,
        sigma_normalized_score_network: ScoreNetwork,
        starting_relative_coordinates: torch.Tensor,
    ):
        """Init method."""
        super().__init__(
            noise_parameters, sampling_parameters, sigma_normalized_score_network
        )

        self._starting_relative_coordinates = starting_relative_coordinates

    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        relative_coordinates = einops.repeat(
            self._starting_relative_coordinates,
            "natoms space -> batch_size natoms space",
            batch_size=number_of_samples,
        )
        return relative_coordinates


class ForcedStartingPointSDEPositionGenerator(ExplodingVarianceSDEPositionGenerator):
    """Forced Starting Point SDE position generator."""
    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: SDESamplingParameters,
        sigma_normalized_score_network: ScoreNetwork,
        starting_relative_coordinates: torch.Tensor,
    ):
        """Init method."""
        super().__init__(
            noise_parameters, sampling_parameters, sigma_normalized_score_network
        )

        self._starting_relative_coordinates = starting_relative_coordinates

    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        relative_coordinates = einops.repeat(
            self._starting_relative_coordinates,
            "natoms space -> batch_size natoms space",
            batch_size=number_of_samples,
        )
        return relative_coordinates


checkpoint_path = (
    "/home/mila/r/rousseab/scratch/experiments/oct2_egnn_1x1x1/run1/"
    "output/last_model/last_model-epoch=049-step=039100.ckpt"
)

spatial_dimension = 3
number_of_atoms = 8
atom_types = np.ones(number_of_atoms, dtype=int)

acell = 5.43

total_time_steps = 1000
number_of_corrector_steps = 10
epsilon = 2.0e-7
noise_parameters = NoiseParameters(
    total_time_steps=total_time_steps,
    corrector_step_epsilon=epsilon,
    sigma_min=0.0001,
    sigma_max=0.2,
)
number_of_samples = 1000

base_sampling_parameters_dict = dict(
    number_of_atoms=number_of_atoms,
    spatial_dimension=spatial_dimension,
    cell_dimensions=[acell, acell, acell],
    number_of_samples=number_of_samples,
)

ode_sampling_parameters = ODESamplingParameters(
    absolute_solver_tolerance=1.0e-5,
    relative_solver_tolerance=1.0e-5,
    **base_sampling_parameters_dict,
)

# Fiddling with SDE is PITA. Also, is there a bug in there?
sde_sampling_parameters = SDESamplingParameters(
    adaptative=False, **base_sampling_parameters_dict
)


langevin_sampling_parameters = PredictorCorrectorSamplingParameters(
    number_of_corrector_steps=number_of_corrector_steps, **base_sampling_parameters_dict
)

device = torch.device("cuda")
if __name__ == "__main__":
    basis_vectors = torch.diag(torch.tensor([acell, acell, acell])).to(device)

    logger.info("Loading checkpoint...")
    pl_model = PositionDiffusionLightningModel.load_from_checkpoint(checkpoint_path)
    pl_model.eval()

    sigma_normalized_score_network = pl_model.sigma_normalized_score_network

    for parameter in sigma_normalized_score_network.parameters():
        parameter.requires_grad_(False)

    equilibrium_relative_coordinates = (
        torch.from_numpy(get_silicon_supercell(supercell_factor=1))
        .to(torch.float32)
        .to(device)
    )

    ode_generator = ForcedStartingPointODEPositionGenerator(
        noise_parameters=noise_parameters,
        sampling_parameters=ode_sampling_parameters,
        sigma_normalized_score_network=sigma_normalized_score_network,
        starting_relative_coordinates=equilibrium_relative_coordinates,
    )

    sde_generator = ForcedStartingPointSDEPositionGenerator(
        noise_parameters=noise_parameters,
        sampling_parameters=sde_sampling_parameters,
        sigma_normalized_score_network=sigma_normalized_score_network,
        starting_relative_coordinates=equilibrium_relative_coordinates,
    )

    langevin_generator = ForcedStartingPointLangevinGenerator(
        noise_parameters=noise_parameters,
        sampling_parameters=langevin_sampling_parameters,
        sigma_normalized_score_network=sigma_normalized_score_network,
        starting_relative_coordinates=equilibrium_relative_coordinates,
    )

    unit_cells = einops.repeat(basis_vectors, "s1 s2 -> b s1 s2", b=number_of_samples)

    ode_samples = ode_generator.sample(
        number_of_samples=number_of_samples, device=device, unit_cell=unit_cells
    )
    sde_samples = sde_generator.sample(
        number_of_samples=number_of_samples, device=device, unit_cell=unit_cells
    )

    langevin_samples = langevin_generator.sample(
        number_of_samples=number_of_samples, device=device, unit_cell=unit_cells
    )

    energy_calculator = EnergyCalculator(
        unit_cell=basis_vectors, number_of_atoms=number_of_atoms
    )

    ode_energies = energy_calculator.compute_oracle_energies(ode_samples)
    sde_energies = energy_calculator.compute_oracle_energies(sde_samples)
    langevin_energies = energy_calculator.compute_oracle_energies(langevin_samples)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Energies of Samples Drawn from Equilibrium Coordinates")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title("Zoom In")
    ax2.set_title("Broad View")

    list_q = np.linspace(0, 1, 101)

    list_energies = [ode_energies, sde_energies, langevin_energies]

    list_colors = ["blue", "red", "green"]

    langevin_label = (
        f"LANGEVIN (time steps = {total_time_steps}, "
        f"corrector steps = {number_of_corrector_steps}, epsilon ={epsilon: 5.2e})"
    )

    list_labels = ["ODE", "SDE", langevin_label]

    for ax in [ax1, ax2]:
        for energies, label in zip(list_energies, list_labels):
            quantiles = np.quantile(energies, list_q)
            ax.plot(100 * list_q, quantiles, "-", label=label)

        ax.fill_between(
            [0, 100],
            y1=-34.6,
            y2=-34.1,
            color="yellow",
            alpha=0.25,
            label="Training Energy Range",
        )

        ax.set_xlabel("Quantile (%)")
        ax.set_ylabel("Energy (eV)")
        ax.set_xlim(-0.1, 100.1)
    ax1.set_ylim(-35, -34.0)
    ax2.legend(loc="upper right", fancybox=True, shadow=True, ncol=1, fontsize=6)

    fig.tight_layout()

    plt.show()

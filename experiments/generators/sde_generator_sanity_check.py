"""SDE Generator Sanity Check.

Check that the SDE generator can solve simple SDEs in 1 dimension with the analytical score as the source of drift.
"""

import einops
import numpy as np
import torch
from matplotlib import pyplot as plt

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.generators.sde_position_generator import (
    ExplodingVarianceSDEPositionGenerator, SDESamplingParameters)
from crystal_diffusion.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetworkParameters, TargetScoreBasedAnalyticalScoreNetwork)
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from experiments.analysis.analytic_score.utils import get_exact_samples
from experiments.generators import GENERATOR_SANITY_CHECK_DIRECTORY

plt.style.use(PLOT_STYLE_PATH)


class DisplacementCalculator:
    """Calculate the displacement distribution."""

    def __init__(self, equilibrium_relative_coordinates: torch.Tensor):
        """Init method."""
        self.equilibrium_relative_coordinates = equilibrium_relative_coordinates

    def compute_displacements(self, batch_relative_coordinates: torch.Tensor) -> np.ndarray:
        """Compute displacements."""
        return (batch_relative_coordinates - equilibrium_relative_coordinates).flatten()


def generate_exact_samples(equilibrium_relative_coordinates: torch.Tensor, sigma_d: float, number_of_samples: int):
    """Generate Gaussian samples about the equilibrium relative coordinates."""
    variance_parameter = sigma_d ** 2

    number_of_atoms, spatial_dimension = equilibrium_relative_coordinates.shape
    nd = number_of_atoms * spatial_dimension

    inverse_covariance = torch.diag(torch.ones(nd)) / variance_parameter
    inverse_covariance = inverse_covariance.reshape(number_of_atoms, spatial_dimension,
                                                    number_of_atoms, spatial_dimension)

    exact_samples = get_exact_samples(equilibrium_relative_coordinates, inverse_covariance, number_of_samples)
    exact_samples = map_relative_coordinates_to_unit_cell(exact_samples)
    return exact_samples


device = torch.device('cpu')

number_of_atoms = 1
spatial_dimension = 1
kmax = 8
acell = 1.
cell_dimensions = [acell]

output_dir = GENERATOR_SANITY_CHECK_DIRECTORY / "figures"
output_dir.mkdir(exist_ok=True)

number_of_samples = 1000

list_sigma_d = [0.1, 0.01, 0.001]
sigma_min = 0.001
total_time_steps = 101

if __name__ == '__main__':

    unit_cell = torch.diag(torch.tensor(cell_dimensions))
    batched_unit_cells = einops.repeat(unit_cell, "d1 d2 -> b d1 d2", b=number_of_samples)

    equilibrium_relative_coordinates = torch.tensor([[0.5]])
    displacement_calculator = DisplacementCalculator(equilibrium_relative_coordinates=equilibrium_relative_coordinates)

    for sigma_d in list_sigma_d:

        exact_samples = generate_exact_samples(equilibrium_relative_coordinates,
                                               sigma_d=sigma_d,
                                               number_of_samples=number_of_samples)

        exact_samples_displacements = displacement_calculator.compute_displacements(exact_samples)

        score_network_parameters = AnalyticalScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            use_permutation_invariance=False,
            kmax=kmax,
            equilibrium_relative_coordinates=equilibrium_relative_coordinates,
            variance_parameter=sigma_d**2)

        sigma_normalized_score_network = TargetScoreBasedAnalyticalScoreNetwork(score_network_parameters)

        sampling_parameters = SDESamplingParameters(method='euler',
                                                    adaptative=False,
                                                    absolute_solver_tolerance=1.0e-7,
                                                    relative_solver_tolerance=1.0e-5,
                                                    number_of_atoms=number_of_atoms,
                                                    spatial_dimension=spatial_dimension,
                                                    number_of_samples=number_of_samples,
                                                    sample_batchsize=number_of_samples,
                                                    cell_dimensions=cell_dimensions,
                                                    record_samples=True)

        noise_parameters = NoiseParameters(total_time_steps=total_time_steps,
                                           sigma_min=sigma_min,
                                           sigma_max=0.5)

        generator = ExplodingVarianceSDEPositionGenerator(noise_parameters=noise_parameters,
                                                          sampling_parameters=sampling_parameters,
                                                          sigma_normalized_score_network=sigma_normalized_score_network)

        generated_samples = generator.sample(number_of_samples,
                                             device=device,
                                             unit_cell=batched_unit_cells)

        sampled_x = generated_samples[:, 0, 0]

        trajectory_data = generator.sample_trajectory_recorder.standardize_data(
            generator.sample_trajectory_recorder.data)

        times = trajectory_data['time']
        x = trajectory_data['relative_coordinates'][:, :, 0, 0]

        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        fig.suptitle("1 Atom in 1D, Analytical Score, SDE solver"
                     f"\n $\\sigma_d$ = {sigma_d:4.3f}, time steps = {total_time_steps}")
        ax1 = fig.add_subplot(121)
        ax1.set_title("Subset of Sampling Trajectories")
        for i in range(50):
            ax1.plot(times, x[i, :], '-', color='gray', alpha=0.5)
        ax1.set_xlim(1, 0)
        ax1.set_xlabel('Diffusion Time')
        ax1.set_ylabel('$x(t)$')

        generated_samples_displacements = displacement_calculator.compute_displacements(generated_samples)

        ax2 = fig.add_subplot(122)
        ax2.set_title("Distribution of Displacements")

        common_params = dict(density=True, bins=101, histtype="stepfilled", alpha=0.25)
        ax2.hist(exact_samples_displacements, **common_params, label='Exact Samples Displacements', color='green')
        ax2.hist(generated_samples_displacements, **common_params, label='Generated Samples Displacements', color='red')

        ax2.set_xlabel('$x - x_0$')
        ax2.set_ylabel('Density')
        ax2.legend(loc='upper right', fancybox=True, shadow=True, ncol=1, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"sde_solution_sigma_d={sigma_d}.png")

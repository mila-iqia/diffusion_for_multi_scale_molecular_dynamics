"""SDE Generator Sanity Check.

Check that the SDE generator can solve simple SDEs in 1 dimension with the analytical score as the source of drift.
This script solves the SDE for various values of the sigma_d parameter and creates plots.
"""
import einops
import torch
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.generators.sde_position_generator import (
    ExplodingVarianceSDEPositionGenerator, SDESamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from experiments.generators import PLOTS_OUTPUT_DIRECTORY
from experiments.generators.utils import (DisplacementCalculator,
                                          generate_exact_samples,
                                          standardize_sde_trajectory_data)

plt.style.use(PLOT_STYLE_PATH)

device = torch.device("cpu")

# Define a very simple situation with a single "atom" in 1D.
number_of_atoms = 1
spatial_dimension = 1
num_atom_types = 1

kmax = 8
acell = 1.0
cell_dimensions = [acell]

number_of_samples = 1000

list_sigma_d = [0.1, 0.01, 0.001]
sigma_min = 0.001
sigma_max = 0.5
total_time_steps = 101

equilibrium_relative_coordinates_list = [[0.5]]
equilibrium_relative_coordinates = torch.tensor(equilibrium_relative_coordinates_list)

if __name__ == "__main__":

    unit_cell = torch.diag(torch.tensor(cell_dimensions))
    batched_unit_cells = einops.repeat(
        unit_cell, "d1 d2 -> b d1 d2", b=number_of_samples
    )

    displacement_calculator = DisplacementCalculator(
        equilibrium_relative_coordinates=equilibrium_relative_coordinates
    )

    for sigma_d in list_sigma_d:

        exact_samples = generate_exact_samples(
            equilibrium_relative_coordinates,
            sigma_d=sigma_d,
            number_of_samples=number_of_samples,
        )

        exact_samples_displacements = displacement_calculator.compute_displacements(
            exact_samples
        )

        score_network_parameters = AnalyticalScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            num_atom_types=num_atom_types,
            kmax=kmax,
            equilibrium_relative_coordinates=equilibrium_relative_coordinates_list,
            sigma_d=sigma_d,
        )

        sigma_normalized_score_network = AnalyticalScoreNetwork(
            score_network_parameters
        )

        sampling_parameters = SDESamplingParameters(
            method="euler",
            adaptive=False,
            absolute_solver_tolerance=1.0e-7,
            relative_solver_tolerance=1.0e-5,
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            num_atom_types=num_atom_types,
            number_of_samples=number_of_samples,
            sample_batchsize=number_of_samples,
            cell_dimensions=cell_dimensions,
            record_samples=True,
        )

        noise_parameters = NoiseParameters(
            total_time_steps=total_time_steps, sigma_min=sigma_min, sigma_max=sigma_max)

        generator = ExplodingVarianceSDEPositionGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=sigma_normalized_score_network,
        )

        # The generated samples are for positions only.
        #   The array has dimensions [number_of_samples,number_of_atoms, spatial_dimension]
        generated_samples = generator.sample(
            number_of_samples, device=device, unit_cell=batched_unit_cells
        )

        sampled_x = generated_samples[:, 0, 0]

        # extract a useful dictionary of the recorded data during sampling.
        trajectory_data = standardize_sde_trajectory_data(generator.sample_trajectory_recorder)

        times = trajectory_data["time"]
        x = trajectory_data["relative_coordinates"][:, :, 0, 0]

        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        fig.suptitle(
            "1 Atom in 1D, Analytical Score, SDE solver"
            f"\n $\\sigma_d$ = {sigma_d:4.3f}, time steps = {total_time_steps}"
        )
        ax1 = fig.add_subplot(121)
        ax1.set_title("Subset of Sampling Trajectories")
        for i in range(50):
            ax1.plot(times, x[i, :], "-", color="gray", alpha=0.5)
        ax1.set_xlim(1, 0)
        ax1.set_xlabel("Diffusion Time")
        ax1.set_ylabel("$x(t)$")

        generated_samples_displacements = displacement_calculator.compute_displacements(
            generated_samples
        )

        ax2 = fig.add_subplot(122)
        ax2.set_title("Distribution of Displacements")

        common_params = dict(density=True, bins=101, histtype="stepfilled", alpha=0.25)
        ax2.hist(
            exact_samples_displacements,
            **common_params,
            label="Exact Samples Displacements",
            color="green",
        )
        ax2.hist(
            generated_samples_displacements,
            **common_params,
            label="Generated Samples Displacements",
            color="red",
        )

        ax2.set_xlabel("$x - x_0$")
        ax2.set_ylabel("Density")
        ax2.legend(loc="upper right", fancybox=True, shadow=True, ncol=1, fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_OUTPUT_DIRECTORY / f"sde_solution_sigma_d={sigma_d}.png")
        plt.show()

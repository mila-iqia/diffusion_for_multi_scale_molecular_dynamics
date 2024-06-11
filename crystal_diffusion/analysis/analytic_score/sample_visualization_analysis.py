import logging

import einops
import matplotlib.pyplot as plt
import torch

from crystal_diffusion import ANALYSIS_RESULTS_DIR
from crystal_diffusion.analysis import PLOT_STYLE_PATH
from crystal_diffusion.analysis.analytic_score.utils import (
    get_exact_samples, get_random_equilibrium_relative_coordinates,
    get_random_inverse_covariance, get_unit_cells)
from crystal_diffusion.generators.predictor_corrector_position_generator import \
    AnnealedLangevinDynamicsGenerator
from crystal_diffusion.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from crystal_diffusion.samplers.variance_sampler import NoiseParameters

logger = logging.getLogger(__name__)

plt.style.use(PLOT_STYLE_PATH)

device = torch.device('cpu')

number_of_atoms = 2
spatial_dimension = 2

kmax = 1

noise_parameters = NoiseParameters(total_time_steps=50, sigma_min=0.001, sigma_max=0.5)

number_of_samples = 1000
number_of_corrector_steps = 1

spring_constant_scale = 1000.

number_of_trials = 3

if __name__ == '__main__':
    torch.manual_seed(23)

    for trial_id in range(number_of_trials):
        equilibrium_relative_coordinates = get_random_equilibrium_relative_coordinates(number_of_atoms,
                                                                                       spatial_dimension)
        inverse_covariance = get_random_inverse_covariance(spring_constant_scale, number_of_atoms, spatial_dimension)
        exact_samples = get_exact_samples(equilibrium_relative_coordinates, inverse_covariance, number_of_samples)

        score_network_parameters = AnalyticalScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            kmax=kmax,
            equilibrium_relative_coordinates=equilibrium_relative_coordinates,
            inverse_covariance=inverse_covariance)

        sigma_normalized_score_network = AnalyticalScoreNetwork(score_network_parameters)

        sampler_parameters = dict(noise_parameters=noise_parameters,
                                  number_of_corrector_steps=number_of_corrector_steps,
                                  number_of_atoms=number_of_atoms,
                                  spatial_dimension=spatial_dimension,
                                  record_samples=False,
                                  positions_require_grad=False)

        pc_sampler = AnnealedLangevinDynamicsGenerator(sigma_normalized_score_network=sigma_normalized_score_network,
                                                       **sampler_parameters)

        unit_cell = get_unit_cells(acell=1., spatial_dimension=spatial_dimension, number_of_samples=number_of_samples)

        samples = pc_sampler.sample(number_of_samples, device=device, unit_cell=unit_cell).detach()

        fig = plt.figure(figsize=(7.5, 7.5))
        fig.suptitle(f'Coordinates Distributions, {number_of_atoms} atoms in {spatial_dimension}D')

        ax1 = fig.add_subplot(111, aspect='equal')

        ax1.vlines([0., 1.], 0., 1., colors='k', linestyles='solid', lw=4)
        ax1.hlines([0., 1.], 0., 1., colors='k', linestyles='solid', lw=4)

        flat_exact_samples = einops.rearrange(exact_samples, "batch n d -> (batch n) d")
        flat_samples = einops.rearrange(samples, "batch n d -> (batch n) d")

        ax1.plot(flat_exact_samples[:, 0], flat_exact_samples[:, 1], 'go',
                 mew=0, alpha=0.25, label='Theoretical Distribution')

        ax1.plot(flat_samples[:, 0], flat_samples[:, 1], 'ro',
                 mew=0, alpha=0.25, label='Sampled Distribution')

        ax1.set_xlabel('X1 Relative Coordinates')
        ax1.set_ylabel('X2 Relative Coordinates')
        ax1.legend(loc='upper right', fancybox=True, shadow=True, ncol=1, fontsize=12)
        fig.tight_layout()

        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        fig.tight_layout()

        fig.savefig(ANALYSIS_RESULTS_DIR / f"analytic_score_visualization_{trial_id}.png")

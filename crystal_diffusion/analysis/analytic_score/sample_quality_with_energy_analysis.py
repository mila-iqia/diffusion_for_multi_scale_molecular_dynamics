"""Analytical score sampling for one atom in 1D.

This module seeks to estimate the "dynamical matrix" for Si 1x1x1 from data.
"""
import logging

import matplotlib.pyplot as plt
import torch

from crystal_diffusion import ANALYSIS_RESULTS_DIR
from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.analysis.analytic_score.utils import (
    get_exact_samples, get_random_equilibrium_relative_coordinates,
    get_random_inverse_covariance, get_samples_harmonic_energy, get_unit_cells)
from crystal_diffusion.models.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from crystal_diffusion.samplers.predictor_corrector_position_sampler import \
    AnnealedLangevinDynamicsSampler
from crystal_diffusion.samplers.variance_sampler import NoiseParameters

logger = logging.getLogger(__name__)

plt.style.use(PLOT_STYLE_PATH)

device = torch.device('cpu')

number_of_atoms = 2
spatial_dimension = 3

kmax = 1

noise_parameters = NoiseParameters(total_time_steps=50, sigma_min=0.001, sigma_max=0.5)

number_of_samples = 10000
number_of_corrector_steps = 5

spring_constant_scale = 1000.

if __name__ == '__main__':
    torch.manual_seed(123)

    equilibrium_relative_coordinates = get_random_equilibrium_relative_coordinates(number_of_atoms, spatial_dimension)
    inverse_covariance = get_random_inverse_covariance(spring_constant_scale, number_of_atoms, spatial_dimension)

    exact_samples = get_exact_samples(equilibrium_relative_coordinates, inverse_covariance, number_of_samples)

    exact_harmonic_energies = get_samples_harmonic_energy(equilibrium_relative_coordinates,
                                                          inverse_covariance,
                                                          exact_samples)

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

    pc_sampler = AnnealedLangevinDynamicsSampler(sigma_normalized_score_network=sigma_normalized_score_network,
                                                 **sampler_parameters)

    unit_cell = get_unit_cells(acell=1., spatial_dimension=spatial_dimension, number_of_samples=number_of_samples)

    samples = pc_sampler.sample(number_of_samples, device=device, unit_cell=unit_cell).detach()

    sampled_harmonic_energies = get_samples_harmonic_energy(equilibrium_relative_coordinates,
                                                            inverse_covariance,
                                                            samples)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f'Energy Distribution, {number_of_atoms} atoms in {spatial_dimension}D')

    common_params = dict(density=True, bins=50, histtype="stepfilled", alpha=0.25)

    ax1 = fig.add_subplot(111)
    ax1.hist(sampled_harmonic_energies, **common_params, label='Sampled Energies', color='red')
    ax1.hist(exact_harmonic_energies, **common_params, label='Theoretical Energies', color='green')

    ax1.set_xlim(xmin=-0.01)
    ax1.set_xlabel('Unitless Harmonic Energy')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper right', fancybox=True, shadow=True, ncol=1, fontsize=12)
    fig.tight_layout()
    fig.savefig(ANALYSIS_RESULTS_DIR / f"harmonic_energy_samples_{spatial_dimension}D.png")
    plt.close(fig)

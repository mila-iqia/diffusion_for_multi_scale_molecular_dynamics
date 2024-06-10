"""Perfect Score ODE sampling.

This little ad hoc experiment explores sampling with an ODE solver, using the 'analytic' score.
It works very well!
"""

import logging

import matplotlib.pyplot as plt
import torch
from einops import einops

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.analysis.analytic_score.utils import (get_exact_samples,
                                                             get_unit_cells)
from crystal_diffusion.generators.ode_position_generator import \
    ExplodingVarianceODEPositionGenerator
from crystal_diffusion.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from crystal_diffusion.samplers.variance_sampler import NoiseParameters

logger = logging.getLogger(__name__)

plt.style.use(PLOT_STYLE_PATH)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

spatial_dimension = 3
number_of_atoms = 2
kmax = 1
spring_constant = 1000.
batch_size = 1000
total_time_steps = 41
if __name__ == '__main__':

    noise_parameters = NoiseParameters(total_time_steps=total_time_steps,
                                       sigma_min=0.001,
                                       sigma_max=0.5)

    equilibrium_relative_coordinates = torch.stack([0.25 * torch.ones(spatial_dimension),
                                                    0.75 * torch.ones(spatial_dimension)])
    inverse_covariance = torch.zeros(number_of_atoms, spatial_dimension, number_of_atoms, spatial_dimension)
    for atom_i in range(number_of_atoms):
        for alpha in range(spatial_dimension):
            inverse_covariance[atom_i, alpha, atom_i, alpha] = spring_constant

    score_network_parameters = AnalyticalScoreNetworkParameters(
        number_of_atoms=number_of_atoms,
        spatial_dimension=spatial_dimension,
        kmax=kmax,
        equilibrium_relative_coordinates=equilibrium_relative_coordinates,
        inverse_covariance=inverse_covariance)

    sigma_normalized_score_network = AnalyticalScoreNetwork(score_network_parameters)

    position_generator = ExplodingVarianceODEPositionGenerator(noise_parameters,
                                                               number_of_atoms,
                                                               spatial_dimension,
                                                               sigma_normalized_score_network,
                                                               record_samples=True)

    times = torch.linspace(0, 1, 1001)
    sigmas = position_generator._get_exploding_variance_sigma(times)
    ode_prefactor = position_generator._get_ode_prefactor(sigmas)

    fig0 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig0.suptitle('ODE parameters')

    ax1 = fig0.add_subplot(121)
    ax2 = fig0.add_subplot(122)
    ax1.set_title('$\\sigma$ Parameter')
    ax2.set_title('$\\gamma$ Parameter')
    ax1.plot(times, sigmas, '-')
    ax2.plot(times, ode_prefactor, '-')

    ax1.set_ylabel('$\\sigma(t)$')
    ax2.set_ylabel('$\\gamma(t)$')
    for ax in [ax1, ax2]:
        ax.set_xlabel('Diffusion Time')
        ax.set_xlim([-0.01, 1.01])

    fig0.tight_layout()
    plt.show()

    unit_cell = get_unit_cells(acell=1., spatial_dimension=spatial_dimension, number_of_samples=batch_size)
    relative_coordinates = position_generator.sample(number_of_samples=batch_size, device=device, unit_cell=unit_cell)

    batch_times = position_generator.sample_trajectory_recorder.data['time'][0]
    batch_relative_coordinates = position_generator.sample_trajectory_recorder.data['relative_coordinates'][0]
    batch_flat_relative_coordinates = einops.rearrange(batch_relative_coordinates, "b t n d -> b t (n d)")

    fig1 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig1.suptitle('ODE Trajectories')

    ax = fig1.add_subplot(111)
    ax.set_xlabel('Diffusion Time')
    ax.set_ylabel('Raw Relative Coordinate')
    ax.yaxis.tick_right()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    time = batch_times[0]  # all time arrays are the same
    for flat_relative_coordinates in batch_flat_relative_coordinates[::20]:
        for i in range(number_of_atoms * spatial_dimension):
            coordinate = flat_relative_coordinates[:, i]
            ax.plot(time, coordinate, '-', color='b', alpha=0.05)

    ax.set_xlim([1.01, -0.01])
    plt.show()

    exact_samples = get_exact_samples(equilibrium_relative_coordinates,
                                      inverse_covariance,
                                      batch_size)

    fig2 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig2.suptitle('Comparing ODE and Expected Marignal Distributions')
    ax1 = fig2.add_subplot(131, aspect='equal')
    ax2 = fig2.add_subplot(132, aspect='equal')
    ax3 = fig2.add_subplot(133, aspect='equal')

    xs = einops.rearrange(relative_coordinates, 'b n d -> (b n) d')
    zs = einops.rearrange(exact_samples, 'b n d -> (b n) d')
    ax1.set_title('XY Projection')
    ax1.plot(xs[:, 0], xs[:, 1], 'ro', alpha=0.5, mew=0, label='ODE Solver')
    ax1.plot(zs[:, 0], zs[:, 1], 'go', alpha=0.05, mew=0, label='Exact Samples')

    ax2.set_title('XZ Projection')
    ax2.plot(xs[:, 0], xs[:, 2], 'ro', alpha=0.5, mew=0, label='ODE Solver')
    ax2.plot(zs[:, 0], zs[:, 2], 'go', alpha=0.05, mew=0, label='Exact Samples')

    ax3.set_title('YZ Projection')
    ax3.plot(xs[:, 1], xs[:, 2], 'ro', alpha=0.5, mew=0, label='ODE Solver')
    ax3.plot(zs[:, 1], zs[:, 2], 'go', alpha=0.05, mew=0, label='Exact Samples')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.vlines(x=[0, 1], ymin=0, ymax=1, color='k', lw=2)
        ax.hlines(y=[0, 1], xmin=0, xmax=1, color='k', lw=2)

    ax1.legend(loc=0)
    fig2.tight_layout()
    plt.show()

    fig3 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    ax1 = fig3.add_subplot(131)
    ax2 = fig3.add_subplot(132)
    ax3 = fig3.add_subplot(133)
    fig3.suptitle("Marginal Distributions of t=0 Samples")

    common_params = dict(histtype='stepfilled', alpha=0.5, bins=50)

    ax1.hist(xs[:, 0], **common_params, facecolor='r', label='ODE solver')
    ax2.hist(xs[:, 1], **common_params, facecolor='r', label='ODE solver')
    ax3.hist(xs[:, 2], **common_params, facecolor='r', label='ODE solver')

    ax1.hist(zs[:, 0], **common_params, facecolor='g', label='Exact')
    ax2.hist(zs[:, 1], **common_params, facecolor='g', label='Exact')
    ax3.hist(zs[:, 2], **common_params, facecolor='g', label='Exact')

    ax1.set_xlabel('X')
    ax2.set_xlabel('Y')
    ax3.set_xlabel('Z')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-0.01, 1.01)

    ax1.legend(loc=0)
    fig3.tight_layout()
    plt.show()

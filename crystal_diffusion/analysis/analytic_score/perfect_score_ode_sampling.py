"""Perfect Score ODE sampling.

This little ad hoc experiment explores sampling with an ODE solver, using the 'analytic' score.
It works very well!
"""

import logging

import matplotlib.pyplot as plt
import torch
import torchode as to
from einops import einops

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.analysis.analytic_score.utils import (get_exact_samples,
                                                             get_unit_cells)
from crystal_diffusion.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from crystal_diffusion.namespace import (NOISE, NOISY_RELATIVE_COORDINATES,
                                         TIME, UNIT_CELL)
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell

logger = logging.getLogger(__name__)

plt.style.use(PLOT_STYLE_PATH)

device = torch.device('cpu')

spatial_dimension = 3
number_of_atoms = 2
kmax = 1
spring_constant = 1000.


if __name__ == '__main__':

    noise_parameters = NoiseParameters(total_time_steps=100,
                                       sigma_min=0.001,
                                       sigma_max=0.5)

    times = torch.linspace(0, 1, 1001)
    sigmas = noise_parameters.sigma_min ** (1.0 - times) * noise_parameters.sigma_max ** times
    ode_prefactor = torch.log(torch.tensor(noise_parameters.sigma_max / noise_parameters.sigma_min)) * sigmas

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(times, sigmas, '-')
    ax2.plot(times, ode_prefactor, '-')

    ax1.set_xlabel('time')
    ax2.set_xlabel('time')
    ax1.set_ylabel('sigma')
    ax2.set_ylabel('gamma')
    plt.show()

    def ode_term_factory(noise_parameters: NoiseParameters,
                         sigma_normalized_score_network: AnalyticalScoreNetwork,
                         number_of_atoms: int,
                         spatial_dimension: int):
        """ODE term factory."""
        sig_min = noise_parameters.sigma_min
        sig_max = noise_parameters.sigma_max

        def ode_term(times: torch.Tensor, flat_relative_coordinates: torch.Tensor) -> torch.Tensor:

            # times has dimension [batch]
            # flat_relative_coordinates has dimensions [batch, number of features]

            batch_size = times.shape[0]
            sigmas = sig_min ** (1.0 - times) * sig_max ** times
            ode_prefactor = torch.log(torch.tensor(sig_max / sig_min)) * sigmas

            unit_cell = get_unit_cells(acell=1., spatial_dimension=spatial_dimension, number_of_samples=batch_size)

            relative_coordinates = einops.rearrange(flat_relative_coordinates,
                                                    "b (n d) -> b n d", n=number_of_atoms, d=spatial_dimension)

            batch = {NOISY_RELATIVE_COORDINATES: map_relative_coordinates_to_unit_cell(relative_coordinates),
                     NOISE: sigmas.unsqueeze(-1),
                     TIME: times.unsqueeze(-1),
                     UNIT_CELL: unit_cell}

            # Shape [batch_size, number of atoms, spatial dimension]
            sigma_normalized_scores = sigma_normalized_score_network(batch)
            flat_sigma_normalized_scores = einops.rearrange(sigma_normalized_scores, "b n d -> b (n d)")

            return -ode_prefactor.unsqueeze(-1) * flat_sigma_normalized_scores

        return ode_term

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

    ode_term = ode_term_factory(noise_parameters, sigma_normalized_score_network, number_of_atoms, spatial_dimension)

    batch_size = 1000
    initial_relative_coordinates = torch.rand(batch_size, number_of_atoms, spatial_dimension)
    y0 = einops.rearrange(initial_relative_coordinates, 'b n d -> b (n d)')

    t0 = 0.0
    tf = 1.0
    n_steps = 41
    times = torch.linspace(tf, t0, n_steps)

    t_eval = einops.repeat(times, 'c -> b c', b=batch_size)

    term = to.ODETerm(ode_term)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)
    jit_solver = torch.compile(solver)

    sol = jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))

    fig1 = plt.figure(figsize=PLEASANT_FIG_SIZE)

    ax = fig1.add_subplot(111)
    for t, y in zip(sol.ts, sol.ys):
        for i in range(y.shape[1]):
            ax.plot(t, y[:, i], '-')

    plt.show()

    exact_samples = get_exact_samples(equilibrium_relative_coordinates,
                                      inverse_covariance,
                                      1000)

    fig2 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    ax1 = fig2.add_subplot(121, aspect='equal')
    ax2 = fig2.add_subplot(222)
    ax3 = fig2.add_subplot(224)
    relative_coordinates = einops.rearrange(sol.ys[:, -1, :],
                                            'b (n d) -> b n d', n=number_of_atoms, d=spatial_dimension)

    relative_coordinates = map_relative_coordinates_to_unit_cell(relative_coordinates)

    common_params = dict(histtype='stepfilled', alpha=0.5, bins=50)

    xs = einops.rearrange(relative_coordinates, 'b n d -> (b n) d')
    ax1.plot(xs[:, 0], xs[:, 1], 'ro', alpha=0.5, mew=0, label='ODE Solver')
    ax2.hist(xs[:, 0], **common_params, facecolor='r', label='ODE solver')
    ax3.hist(xs[:, 1], **common_params, facecolor='r', label='ODE solver')

    zs = einops.rearrange(exact_samples, 'b n d -> (b n) d')
    ax1.plot(zs[:, 0], zs[:, 1], 'go', alpha=0.05, mew=0, label='Exact Samples')
    ax2.hist(zs[:, 0], **common_params, facecolor='g', label='Exact')
    ax3.hist(zs[:, 1], **common_params, facecolor='g', label='Exact')

    ax1.set_xlim(-0.01, 1.01)
    ax1.set_ylim(-0.01, 1.01)

    ax1.vlines(x=[0, 1], ymin=0, ymax=1, color='k', lw=2)
    ax1.hlines(y=[0, 1], xmin=0, xmax=1, color='k', lw=2)
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    ax3.legend(loc=0)
    plt.show()

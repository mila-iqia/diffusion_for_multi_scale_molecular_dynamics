"""Analytical score sampling for one atom in 1D.

This module seeks to estimate the "dynamical matrix" for Si 1x1x1 from data.
"""
import logging

import matplotlib.pyplot as plt
import torch
from einops import einops

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.analysis.analytic_score import \
    ANALYTIC_SCORE_RESULTS_DIR
from crystal_diffusion.analysis.analytic_score.utils import get_unit_cells
from crystal_diffusion.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from crystal_diffusion.namespace import (NOISE, NOISY_RELATIVE_COORDINATES,
                                         TIME, UNIT_CELL)

logger = logging.getLogger(__name__)

plt.style.use(PLOT_STYLE_PATH)

device = torch.device('cpu')

number_of_atoms = 1
spatial_dimension = 1

variance_parameter = 1. / 250.

x_eq = 0.5

if __name__ == '__main__':

    grid = torch.linspace(0., 1., 201)[:-1]
    batch_size = len(grid)
    relative_coordinates = einops.repeat(grid, 'b -> b n d', n=number_of_atoms, d=spatial_dimension)
    times = torch.rand(batch_size, 1)  # doesn't matter. Not used by analytical score.
    unit_cell = get_unit_cells(acell=1., spatial_dimension=spatial_dimension, number_of_samples=batch_size)
    batch = {NOISY_RELATIVE_COORDINATES: relative_coordinates, TIME: times, UNIT_CELL: unit_cell}

    equilibrium_relative_coordinates = torch.tensor([[x_eq]])

    kmax = 3
    score_network_parameters = AnalyticalScoreNetworkParameters(
        number_of_atoms=number_of_atoms,
        spatial_dimension=spatial_dimension,
        kmax=kmax,
        equilibrium_relative_coordinates=equilibrium_relative_coordinates,
        variance_parameter=variance_parameter)
    sigma_normalized_score_network = AnalyticalScoreNetwork(score_network_parameters)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f'Sigma normalized Score, 1 atom in 1D\n Equilibrium Position: ({x_eq}), '
                 f'$\\sigma_d^2$ = {variance_parameter}, kmax = {kmax}')

    ax1 = fig.add_subplot(111)

    for sigma in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
        batch[NOISE] = sigma * torch.ones(batch_size, 1)
        sigma_normalized_scores = sigma_normalized_score_network(batch).detach().flatten()
        x = relative_coordinates.detach().flatten()

        ax1.plot(x, sigma_normalized_scores, label=f'$\\sigma$ = {sigma}')

    ax1.set_xlabel('Relative Coordinates')
    ax1.set_ylabel('$\\sigma \\times$ Score')
    ax1.legend(loc='upper right', fancybox=True, shadow=True, ncol=1, fontsize=9)
    ax1.set_xlim([-0.01, 1.01])
    fig.tight_layout()
    fig.savefig(ANALYTIC_SCORE_RESULTS_DIR / "analytical_score_network_1D.png")
    plt.close(fig)

    fig2 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig2.suptitle(f'Sigma normalized Score, 1 atom in 1D\n Equilibrium Position: ({x_eq}), '
                  f'$\\sigma_d^2$ = {variance_parameter}')

    ax2 = fig2.add_subplot(131)
    ax3 = fig2.add_subplot(132)
    ax4 = fig2.add_subplot(133)

    lw = 6
    x = relative_coordinates.detach().flatten()
    for kmax in [0, 1, 2, 3, 4]:
        score_network_parameters = AnalyticalScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            kmax=kmax,
            equilibrium_relative_coordinates=equilibrium_relative_coordinates,
            variance_parameter=variance_parameter)
        sigma_normalized_score_network = AnalyticalScoreNetwork(score_network_parameters)
        for sigma, ax in zip([0.001, 0.1, 0.5], [ax2, ax3, ax4]):
            batch[NOISE] = sigma * torch.ones(batch_size, 1)
            sigma_normalized_scores = sigma_normalized_score_network(batch).detach().flatten()

            ax.plot(x, sigma_normalized_scores, label=f'kmax = {kmax}', lw=lw)
        lw = 0.5 * lw

    ax2.legend(loc='upper right', fancybox=True, shadow=True, ncol=1, fontsize=9)
    for sigma, ax in zip([0.001, 0.1, 0.5], [ax2, ax3, ax4]):
        ax.set_ylabel('$\\sigma \\times$ Score')
        ax.set_xlim([-0.01, 1.01])
        ax.set_title(f"$\\sigma$={sigma}")
    fig2.tight_layout()
    fig2.savefig(ANALYTIC_SCORE_RESULTS_DIR / "analytical_score_network_1D_convergence.png")
    plt.close(fig2)

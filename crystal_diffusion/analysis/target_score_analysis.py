"""Target score analysis.

This script computes and plots the target score for various values of sigma, showing
that the 'smart' implementation converges quickly and is equal to the expected brute force value.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from crystal_diffusion import ANALYSIS_RESULTS_DIR
from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.score.wrapped_gaussian_score import (
    SIGMA_THRESHOLD, get_sigma_normalized_score,
    get_sigma_normalized_score_brute_force)

plt.style.use(PLOT_STYLE_PATH)

if __name__ == '__main__':

    list_u = np.linspace(0, 1, 101)[:-1]
    relative_positions = torch.from_numpy(list_u)

    # A first figure to compare the "smart" and the "brute force" results
    fig1 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig1.suptitle("Smart vs. brute force scores")

    kmax = 4
    ax1 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)

    for sigma_factor, color in zip([0.1, 1., 2.], ['r', 'g', 'b']):
        sigma = sigma_factor * SIGMA_THRESHOLD

        sigmas = torch.ones_like(relative_positions) * sigma
        list_scores_brute = np.array([get_sigma_normalized_score_brute_force(u, sigma) for u in list_u])
        list_scores = get_sigma_normalized_score(relative_positions, sigmas, kmax=kmax).numpy()
        error = list_scores - list_scores_brute

        ax1.plot(list_u, list_scores_brute, '--', c=color, lw=4, label='brute force')
        ax1.plot(list_u, list_scores, '-', c=color, lw=2, label='smart')

        ax2.plot(list_u, error, '-', c=color, label=f'$\\sigma$ = {sigma_factor} $\\sigma_{{th}}$')

    ax1.set_ylabel('$\\sigma^2\\times S$')
    ax2.set_xlabel('u')
    ax2.set_ylabel('Error')

    for ax in [ax1, ax2]:
        ax.set_xlabel('u')
        ax.set_xlim([0, 1])
        ax.legend(loc=0)

    fig1.tight_layout()
    fig1.savefig(ANALYSIS_RESULTS_DIR.joinpath("score_convergence_with_sigma.png"))

    # A second figure to show convergence with kmax
    fig2 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig2.suptitle("Convergence with kmax")

    ax3 = fig2.add_subplot(121)
    ax4 = fig2.add_subplot(122)

    sigma_factors = torch.linspace(0.001, 4., 40).to(torch.double)
    sigmas = sigma_factors * SIGMA_THRESHOLD

    u = 0.6
    relative_positions = torch.ones_like(sigmas).to(torch.double) * u

    ms = 8
    for kmax, color in zip([1, 2, 3, 4, 5], ['y', 'r', 'g', 'b', 'k']):
        list_scores = get_sigma_normalized_score(relative_positions, sigmas, kmax=kmax).numpy()
        ax3.semilogy(sigma_factors, list_scores, 'o-',
                     ms=ms, c=color, lw=2, alpha=0.25, label=f'kmax = {kmax}')

        list_scores_brute = np.array([
            get_sigma_normalized_score_brute_force(u, sigma, kmax=4 * kmax) for sigma in sigmas])
        ax4.semilogy(sigma_factors, list_scores_brute, 'o-',
                     ms=ms, c=color, lw=2, alpha=0.25, label=f'kmax = {4 * kmax}')

        ms = 0.75 * ms

    for ax in [ax3, ax4]:
        ax.set_xlabel('$\\sigma$ ($\\sigma_{th}$)')
        ax.set_ylabel(f'$\\sigma^2 \\times S(u={u})$')
        ax.set_xlim([0., 1.1 * sigma_factors.max()])
        ax.legend(loc=0)

    ax3.set_title("Smart implementation")
    ax4.set_title("Brute force implementation")
    fig2.tight_layout()
    fig2.savefig(ANALYSIS_RESULTS_DIR.joinpath("score_convergence_with_k.png"))

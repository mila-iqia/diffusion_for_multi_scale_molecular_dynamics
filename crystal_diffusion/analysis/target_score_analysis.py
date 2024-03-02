"""Target score analysis.

This script computes and plots the target score for various values of sigma, showing
that the 'smart' implementation converges quickly and is equal to the expected brute force value.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.score.wrapped_gaussian_score import get_expected_sigma_normalized_score_brute_force, \
    SIGMA_THRESHOLD, get_sigma_normalized_score

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
        list_scores_brute = np.array([get_expected_sigma_normalized_score_brute_force(u, sigma) for u in list_u])
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

    # A second figure to show convergence with kmax
    fig2 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig2.suptitle("Convergence with kmax")

    ax3 = fig2.add_subplot(111)

    sigmas = torch.linspace(0.01, 2., 20) * SIGMA_THRESHOLD

    u = 0.6
    relative_positions = torch.ones_like(sigmas) * u

    ms = 8
    for kmax, color in zip([1, 2, 3, 4], ['r', 'g', 'b', 'k']):
        list_scores = get_sigma_normalized_score(relative_positions, sigmas, kmax=kmax).numpy()
        ax3.plot(sigmas, list_scores, 'o-', ms=ms, c=color, lw=2, alpha=0.25, label=f'kmax = {kmax}')
        ms = 0.75 * ms

    ax3.set_xlabel('$\\sigma$')
    ax3.set_ylabel(f'$\\sigma^2 \\times S(u={u})$')
    ax3.set_xlim([0., 1.1 * sigma.max()])
    ax3.legend(loc=0)

    plt.show()

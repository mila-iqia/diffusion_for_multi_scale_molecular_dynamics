import matplotlib.pyplot as plt
import numpy as np
import torch

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.score.wrapped_gaussian_score import get_expected_sigma_normalized_score_brute_force, \
    SIGMA_THRESHOLD, get_sigma_normalized_score

plt.style.use(PLOT_STYLE_PATH)

if __name__ == '__main__':

    kmax = 2
    list_u = np.linspace(0, 1, 101)[:-1]
    sigma_factor = 0.1
    sigma = sigma_factor * SIGMA_THRESHOLD

    list_scores_brute = np.array([get_expected_sigma_normalized_score_brute_force(u, sigma) for u in list_u])

    relative_positions = torch.from_numpy(list_u)
    sigmas = torch.ones_like(relative_positions) * sigma
    list_scores = get_sigma_normalized_score(relative_positions, sigmas, kmax=kmax).numpy()
    error = list_scores - list_scores_brute

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.set_title(f'$\\sigma$ = {sigma_factor} $\\sigma_{{th}}$')
    ax1.plot(list_u, list_scores_brute, 'b--', lw=4, label='brute force')
    ax1.plot(list_u, list_scores, 'g-', lw=2, label='smart')

    ax2.plot(list_u, error, 'r-')

    ax1.set_ylabel('$\\sigma^2\\times S$')

    ax2.set_xlabel('u')
    ax2.set_ylabel('Error')

    ax1.legend(loc=0)

    for ax in [ax1, ax2]:
        ax.set_xlabel('u')
        ax.set_xlim([0, 1])

    fig.tight_layout()

    plt.show()

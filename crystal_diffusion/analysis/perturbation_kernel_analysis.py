"""Perturbation Kernel Analysis.

This script computes and plots the perturbation kernel K for various values of sigma, showing
the behavior for different normalization, sigma K and sigma^2 K.
"""
import matplotlib.pyplot as plt
import torch

from crystal_diffusion import ANALYSIS_RESULTS_DIR
from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.score.wrapped_gaussian_score import \
    get_sigma_normalized_score

plt.style.use(PLOT_STYLE_PATH)

sigma_max = 0.5

if __name__ == '__main__':

    coordinates = torch.linspace(start=0, end=1, steps=1001)[:-1]
    sigmas = torch.linspace(start=0, end=sigma_max, steps=101)[1:]

    X, SIG = torch.meshgrid(coordinates, sigmas, indexing='xy')

    # Avoid non-contiguous bjorks
    X = X.clone()
    SIG = SIG.clone()

    sigma_normalized_scores = get_sigma_normalized_score(X, SIG, kmax=4).numpy()
    sigma2_normalized_scores = SIG * sigma_normalized_scores

    # A first figure to compare the "smart" and the "brute force" results
    fig_size = (1.25 * PLEASANT_FIG_SIZE[0], 1.25 * PLEASANT_FIG_SIZE[1])
    fig = plt.figure(figsize=fig_size)
    fig.suptitle("Different Normalizations of the Perturbation Kernel, $K(x, \\sigma)$")

    left, right, bottom, top = 0, 1., 0., sigma_max
    extent = [left, right, bottom, top]

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)

    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    label1 = "$\\sigma \\times K(x, \\sigma)$"
    label2 = "$\\sigma^2 \\times K(x, \\sigma)$"

    for label, scores, ax in zip([label1, label2], [sigma_normalized_scores, sigma2_normalized_scores], [ax1, ax3]):

        ax.set_title(label)
        image = ax.imshow(scores, origin="lower", cmap='jet', extent=extent)
        ax.set_xlabel("x")
        ax.set_ylabel("$\\sigma$")
        _ = fig.colorbar(image, shrink=0.75, ax=ax)

    for index in [0, 19, 39, 99]:
        sigma = sigmas[index]
        for ax, scores in zip([ax2, ax4], [sigma_normalized_scores, sigma2_normalized_scores]):
            ax.plot(coordinates, scores[index], ls='-', lw=2, label=f'$\\sigma = {sigma:4.3f}$')

    for ax in [ax2, ax4]:
        ax.legend(loc=0)
        ax.set_xlim([-0.05, 1.05])
        ax.set_xlabel("x")
    ax2.set_ylabel("$\\sigma \\times K(x, \\sigma)$")
    ax4.set_ylabel("$\\sigma^2 \\times K(x, \\sigma)$")

    fig.tight_layout()
    fig.savefig(ANALYSIS_RESULTS_DIR.joinpath("perturbation_kernel_with_different_normalizations.png"))

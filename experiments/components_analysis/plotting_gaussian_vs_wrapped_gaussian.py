"""Plotting Gaussians vs. Wrapped Gaussians.

This script plots a comparison between a 1D Gaussian centered at 0.5 and the sum of its periodic images.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from experiments.components_analysis import PLOTS_OUTPUT_DIRECTORY

plt.style.use(PLOT_STYLE_PATH)

sigma_max = 0.5

list_sigmas = [0.1, 0.25, 0.5]
list_colors = ["r", "b", "g"]

imin = -1
imax = 2
if __name__ == "__main__":

    x = np.linspace(imin, imax, 1001)

    x_in_cell = np.linspace(0, 1, 1001)

    list_locs = np.arange(-8, 9) + 0.5

    for sigma, color in zip(list_sigmas, list_colors):
        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        fig.suptitle(f"Gaussian vs. Wrapped Gaussian for $\\sigma = {sigma}$")
        ax = fig.add_subplot(111)
        in_cell_gaussian = norm(loc=0.5, scale=sigma).pdf(x_in_cell)
        ymax1 = np.max(in_cell_gaussian) + 0.1
        ax.plot(x_in_cell, in_cell_gaussian, lw=2, color=color, label="Gaussian")

        wrapped_gaussians = np.zeros(len(x))
        label = "Periodic Images"
        for loc in list_locs:
            gaussian = norm(loc=loc, scale=sigma).pdf(x)
            wrapped_gaussians += gaussian
            ax.plot(x, gaussian, color=color, lw=1, ls="-", alpha=0.5, label=label)
            label = "__nolabel__"

        ymax2 = np.max(wrapped_gaussians) + 0.1
        ymax = np.max([ymax1, ymax2])

        ax.plot(
            x, wrapped_gaussians, color=color, lw=2, ls="--", label="Wrapped Gaussian"
        )

        ax.vlines(np.arange(imin, imax + 1), 0, ymax, color="k")

        ax.fill_betweenx(
            [0, ymax], x1=0, x2=1, color="grey", alpha=0.25, label="Unit Cell"
        )

        ax.set_ylim(0, ymax)
        ax.set_xlim(imin - 0.1, imax + 0.1)
        ax.legend(loc=0)

        fig.tight_layout()
        fig.savefig(
            PLOTS_OUTPUT_DIRECTORY.joinpath(
                f"Gaussian_vs_Wrapped_Gaussian_sigma={sigma}.png"
            )
        )
        plt.show()

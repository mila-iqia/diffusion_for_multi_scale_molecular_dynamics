"""Plotting Diffusion for Poster.

This script generates images that show how diffusion works at a conceptual level. It was used to create
a scientific poster. This is not real data or a real input to any model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOTS_OUTPUT_DIRECTORY = Path(__file__).parent / "images"
PLOTS_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    x_max = 1000
    t_max = 1500

    x = np.linspace(0, 1, x_max)
    t = np.linspace(0, 1, t_max)

    T, X = np.meshgrid(t, x)

    SIGMA = 0.5 * T + 0.001

    P = np.zeros([x_max, t_max])
    for x0 in [0.25, 0.5, 0.85]:
        P += np.exp(-((X - x0) ** 2) / SIGMA**2)

    noise = np.random.randn(100)
    list_times = np.linspace(0, t_max - 1, 100)

    noise_scale = 0.05 * list_times / t_max
    list_x = (0.5 + noise_scale * np.cumsum(noise)) * x_max

    fig1 = plt.figure(figsize=(6.5, 4))
    fig2 = plt.figure(figsize=(6.5, 4))

    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax1.imshow(P)
    ax2.imshow(P[:, ::-1])

    ax2.plot(list_times[::-1], list_x, "r-", lw=3)
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])

    fig1.tight_layout()
    fig2.tight_layout()

    fig1.savefig(PLOTS_OUTPUT_DIRECTORY / "forward_diffusion.png")
    fig2.savefig(PLOTS_OUTPUT_DIRECTORY / "backward_diffusion.png")

    plt.show()

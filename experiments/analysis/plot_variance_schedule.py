"""Plot Variance Schedule.

This script computes and plots the variance schedule used to noise and denoise
the relative positions.
"""

import matplotlib.pyplot as plt
import torch

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from diffusion_for_multi_scale_molecular_dynamics.score.wrapped_gaussian_score import \
    get_coordinates_sigma_normalized_score
from experiments.analysis import PLOTS_OUTPUT_DIRECTORY

plt.style.use(PLOT_STYLE_PATH)

if __name__ == "__main__":

    noise_parameters = NoiseParameters(total_time_steps=1000)
    noise_scheduler = NoiseScheduler(noise_parameters=noise_parameters, num_classes=1)

    noise, langevin_dynamics = noise_scheduler.get_all_sampling_parameters()

    fig1 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig1.suptitle("Noise Schedule")

    ax1 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)

    ax1.plot(noise.time, noise.sigma, "-", c="b", lw=2, label="$\\sigma(t)$")
    ax1.plot(noise.time, noise.g, "-", c="g", lw=2, label="$g(t)$")

    shifted_time = torch.cat([torch.tensor([0]), noise.time[:-1]])
    ax1.plot(
        shifted_time,
        langevin_dynamics.epsilon,
        "-",
        c="r",
        lw=2,
        label="$\\epsilon(t)$",
    )
    ax1.legend(loc=0)

    ax1.set_xlabel("time")
    ax1.set_xlim([-0.01, 1.01])

    ax1.set_title("$\\sigma, g, \\epsilon$ schedules")

    relative_positions = torch.linspace(0, 1, 101)[:-1]

    kmax = 4
    indices = torch.tensor([1, 250, 750, 999])

    times = noise.time.take(indices)
    sigmas = noise.sigma.take(indices)
    gs_squared = noise.g_squared.take(indices)

    for t, sigma in zip(times, sigmas):
        target_sigma_normalized_scores = get_coordinates_sigma_normalized_score(
            relative_positions, torch.ones_like(relative_positions) * sigma, kmax=kmax
        )
        ax2.plot(
            relative_positions, target_sigma_normalized_scores, label=f"t = {t:3.2f}"
        )

    ax2.set_title("Target Normalized Score")
    ax2.set_xlabel("relative position, u")
    ax2.set_ylabel("$\\sigma(t) \\times S(u, t)$")
    ax2.legend(loc=0)
    ax2.set_xlim([-0.01, 1.01])

    fig1.tight_layout()
    fig1.savefig(PLOTS_OUTPUT_DIRECTORY / "variance_schedule.png")
    plt.show()

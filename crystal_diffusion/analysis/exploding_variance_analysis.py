"""Exploding variance analysis.

This script computes and plots the variance schedule used to noise and denoise
the relative positions.
"""
import matplotlib.pyplot as plt
import torch

from crystal_diffusion import ANALYSIS_RESULTS_DIR
from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.samplers.time_sampler import TimeParameters, TimeSampler
from crystal_diffusion.samplers.variance_sampler import (
    ExplodingVarianceSampler, VarianceParameters)
from crystal_diffusion.score.wrapped_gaussian_score import \
    get_sigma_normalized_score

plt.style.use(PLOT_STYLE_PATH)

if __name__ == '__main__':

    variance_parameters = VarianceParameters()
    time_parameters = TimeParameters(total_time_steps=1000)

    time_sampler = TimeSampler(time_parameters=time_parameters)
    variance_sampler = ExplodingVarianceSampler(variance_parameters=variance_parameters,
                                                time_sampler=time_sampler)

    indices = torch.arange(time_parameters.total_time_steps)
    times = time_sampler.get_time_steps(indices)
    sigmas = torch.sqrt(variance_sampler.get_variances(indices))
    gs = torch.sqrt(variance_sampler.get_g_squared_factors(indices[1:]))

    # A first figure to compare the "smart" and the "brute force" results
    fig1 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig1.suptitle("Noise Schedule")

    ax1 = fig1.add_subplot(221)
    ax2 = fig1.add_subplot(223)
    ax3 = fig1.add_subplot(122)

    ax1.plot(times, sigmas, '-', c='k', lw=2)
    ax2.plot(times[1:], gs, '-', c='k', lw=2)

    ax1.set_ylabel('$\\sigma(t)$')
    ax2.set_ylabel('$g(t)$')

    for ax in [ax1, ax2]:
        ax.set_xlabel('time')
        ax.set_xlim([-0.01, 1.01])

    ax1.set_title("$\\sigma$ schedule")
    ax2.set_title("g schedule")

    relative_positions = torch.linspace(0, 1, 101)[:-1]

    kmax = 4
    indices = torch.tensor([1, 250, 750, 999])
    times = time_sampler.get_time_steps(indices)
    sigmas = torch.sqrt(variance_sampler.get_variances(indices))
    gs_squared = variance_sampler.get_g_squared_factors(indices)

    for t, sigma in zip(times, sigmas):
        target_scores = get_sigma_normalized_score(relative_positions,
                                                   torch.ones_like(relative_positions) * sigma,
                                                   kmax=kmax)
        ax3.plot(relative_positions, sigma * target_scores, label=f"t = {t:3.2f}")

    ax3.set_title("Target Noise")
    ax3.set_xlabel("relative position, u")
    ax3.set_ylabel("$\\sigma(t) \\times S(u, t)$")
    ax3.legend(loc=0)
    ax3.set_xlim([-0.01, 1.01])

    fig1.tight_layout()

    fig1.savefig(ANALYSIS_RESULTS_DIR.joinpath("variance_schedule.png"))

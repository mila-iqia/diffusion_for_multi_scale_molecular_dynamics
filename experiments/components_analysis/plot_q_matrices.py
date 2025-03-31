"""Plotting Q matrices.

This script plots the relevant terms making up the Q matrices, used for atom type diffusion.
"""

from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from experiments.components_analysis import PLOTS_OUTPUT_DIRECTORY

plt.style.use(PLOT_STYLE_PATH)

num_classes = 3

if __name__ == '__main__':

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)

    fig.suptitle("Transition Probabilities")
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    for total_time_steps in [1000, 100, 10]:
        noise_parameters = NoiseParameters(total_time_steps=total_time_steps)
        sampler = NoiseScheduler(noise_parameters, num_classes=num_classes)
        noise, _ = sampler.get_all_sampling_parameters()
        times = noise.time
        indices = noise.indices
        q_matrices = noise.q_matrix
        q_bar_matrices = noise.q_bar_matrix

        betas = q_matrices[:, 0, -1]
        beta_bars = q_bar_matrices[:, 0, -1]
        ratio = beta_bars[:-1] / beta_bars[1:]
        ax1.plot(times, betas, label=f'T = {total_time_steps}')
        ax2.plot(times, beta_bars, label=f'T = {total_time_steps}')
        ax3.plot(times[1:], ratio, label=f'T = {total_time_steps}')

    ax1.set_ylabel(r'$\beta_t$')
    ax2.set_ylabel(r'$\bar\beta_{t}$')
    ax3.set_ylabel(r'$\frac{\bar\beta_{t-1}}{\bar\beta_{t}}$')
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(r'$\frac{t}{T}$')
        ax.legend(loc=0)
        ax.set_xlim(times[-1] + 0.1, times[0] - 0.1)

    fig.tight_layout()
    fig.savefig(PLOTS_OUTPUT_DIRECTORY / "q_matrices.png")
    plt.show()

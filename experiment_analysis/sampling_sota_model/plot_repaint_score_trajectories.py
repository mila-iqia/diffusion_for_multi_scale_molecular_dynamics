"""Plotting the score trajectories of repaint structure, with SOTA model."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import einops

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.utils.logging_utils import setup_analysis_logger

logger = logging.getLogger(__name__)
setup_analysis_logger()


base_path = Path("/Users/bruno/courtois/draw_sota_samples/repaint/")
data_pickle_path = base_path / "repaint_trajectories.pkl"
energy_pickle_path = base_path / "energies.pt"


unconstrained_energy_pickle_path = Path("/Users/bruno/courtois/draw_sota_samples/1x1x1/LANGEVIN/energy_samples.pt")

number_of_constrained_coordinates = 9

plt.style.use(PLOT_STYLE_PATH)

if __name__ == '__main__':

    logger.info("Extracting data artifacts")
    with open(data_pickle_path, 'rb') as fd:
        recorded_data = torch.load(fd, map_location=torch.device('cpu'))

    repaint_energies = torch.load(energy_pickle_path)
    unconstrained_energies = torch.load(unconstrained_energy_pickle_path)

    # ==============================================================
    logger.info("Plotting energy distributions")
    fig0 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig0.suptitle('Comparing Unconstrained and Repaint Sample Energy Distributions')

    common_params = dict(density=True, bins=80, histtype="stepfilled", alpha=0.25)

    ax1 = fig0.add_subplot(111)
    ax1.hist(unconstrained_energies, **common_params, label='Unconstrained Energies', color='green')
    ax1.hist(repaint_energies, **common_params, label='Repaint Energies', color='red')

    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper right', fancybox=True, shadow=True, ncol=1, fontsize=12)
    ax1.set_yscale('log')
    fig0.tight_layout()
    fig0.savefig(base_path / "comparing_energy_distributions_unconstrained_vs_repaint.png")
    plt.close(fig0)

    sampling_times = recorded_data['time']
    relative_coordinates = recorded_data['relative_coordinates']
    batch_flat_relative_coordinates = einops.rearrange(relative_coordinates, "b t n d -> b t (n d)")

    normalized_scores = recorded_data['normalized_scores']
    batch_flat_normalized_scores = einops.rearrange(normalized_scores, "b t n d -> b t (n d)")

    logger.info("Plotting scores along trajectories")
    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle('Root Mean Square Normalized Scores Along Sample Trajectories')
    rms_norm_score = (batch_flat_normalized_scores ** 2).mean(dim=-1).sqrt().numpy()

    constrained_rms_norm_score = (
        (batch_flat_normalized_scores[:, :, :number_of_constrained_coordinates] ** 2).mean(dim=-1).sqrt().numpy())

    free_rms_norm_score = (
        (batch_flat_normalized_scores[:, :, number_of_constrained_coordinates:] ** 2).mean(dim=-1).sqrt().numpy())

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    for ax, scores in zip([ax1, ax2, ax3, ax4],
                          [rms_norm_score, rms_norm_score, constrained_rms_norm_score, free_rms_norm_score]):
        for y in scores[::10]:
            ax.plot(sampling_times, y, '-', color='gray', alpha=0.2, label='__nolabel__')

    list_quantiles = [0.0, 0.10, 0.5, 1.0]
    list_colors = ['green', 'yellow', 'orange', 'red']

    for q, c in zip(list_quantiles, list_colors):
        energy_quantile = np.quantile(repaint_energies, q)
        idx = np.argmin(np.abs(repaint_energies - energy_quantile))
        e = repaint_energies[idx]
        for ax, scores in zip([ax1, ax2, ax3, ax4],
                              [rms_norm_score, rms_norm_score, constrained_rms_norm_score, free_rms_norm_score]):
            ax.plot(sampling_times, scores[idx], '-',
                    color=c, alpha=1., label=f'Q = {100 * q:2.0f}%, Energy: {e:5.1f}')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Diffusion Time')
        ax.set_ylabel(r'$\sqrt{\langle (\sigma(t) S_{\theta} )^2 \rangle}$')
        ax.set_xlim(1, 0)

    for ax in [ax2, ax3, ax4]:
        ax.set_yscale('log')

    ax1.set_title("All Coordinates (Not Log Scale)")
    ax1.set_ylim(ymin=0.)

    ax2.set_title("All Coordinates")
    ax3.set_title("Restrained Coordinates")
    ax4.set_title("Free Coordinates")

    ax1.legend(loc=0, fontsize=6)
    fig.tight_layout()
    fig.savefig(base_path / "sampling_score_trajectories_repaint.png")
    plt.close(fig)

    logger.info("Done!")

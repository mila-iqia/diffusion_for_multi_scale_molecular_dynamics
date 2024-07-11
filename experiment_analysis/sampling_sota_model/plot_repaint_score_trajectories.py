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


plt.style.use(PLOT_STYLE_PATH)

if __name__ == '__main__':

    logger.info("Extracting data artifacts")
    with open(data_pickle_path, 'rb') as fd:
        recorded_data = torch.load(fd, map_location=torch.device('cpu'))

    energies = torch.load(energy_pickle_path)

    sampling_times = recorded_data['time']
    relative_coordinates = recorded_data['relative_coordinates']
    batch_flat_relative_coordinates = einops.rearrange(relative_coordinates, "b t n d -> b t (n d)")

    normalized_scores = recorded_data['normalized_scores']
    batch_flat_normalized_scores = einops.rearrange(normalized_scores, "b t n d -> b t (n d)")

    logger.info("Plotting scores along trajectories")
    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle('Root Mean Square Normalized Scores Along Sample Trajectories')
    rms_norm_score = (batch_flat_normalized_scores ** 2).mean(dim=-1).sqrt().numpy()

    ax1 = fig.add_subplot(111)
    for y in rms_norm_score[::10]:
        ax1.plot(sampling_times, y, '-', color='gray', alpha=0.2, label='__nolabel__')

    list_quantiles = [0.0, 0.10, 0.5, 1.0]
    list_colors = ['green', 'yellow', 'orange', 'red']

    for q, c in zip(list_quantiles, list_colors):
        energy_quantile = np.quantile(energies, q)
        idx = np.argmin(np.abs(energies - energy_quantile))
        e = energies[idx]
        ax1.plot(sampling_times, rms_norm_score[idx], '-',
                 color=c, alpha=1., label=f'{100 * q:2.0f}% Percentile Energy: {e:5.1f}')

    ax1.set_xlabel('Diffusion Time')
    ax1.set_ylabel(r'$\sqrt{\langle (\sigma(t) S_{\theta} )^2 \rangle}$')
    ax1.legend(loc=0)
    ax1.set_xlim(1, 0)

    fig.savefig(base_path / "sampling_score_trajectories_repaint.png")
    plt.close(fig)

    logger.info("Done!")

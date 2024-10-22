"""Plotting the score trajectories of repaint structure, with the Analytical model."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import einops

from diffusion_for_multi_scale_molecular_dynamics import ANALYSIS_RESULTS_DIR
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    setup_analysis_logger
from experiments.analysis.analytic_score.utils import (
    get_exact_samples, get_samples_harmonic_energy, get_silicon_supercell)

logger = logging.getLogger(__name__)
setup_analysis_logger()


base_path = ANALYSIS_RESULTS_DIR / "ANALYTIC_SCORE/REPAINT"
data_pickle_path = base_path / "repaint_trajectories.pkl"
energy_pickle_path = base_path / "harmonic_energies.pt"

plt.style.use(PLOT_STYLE_PATH)

number_of_constrained_coordinates = 3 * 3


supercell_factor = 1
variance_parameter = 0.001 / supercell_factor
number_of_samples = 1000

number_of_atoms, spatial_dimension = 8, 3
nd = number_of_atoms * spatial_dimension

if __name__ == "__main__":

    inverse_covariance = torch.diag(torch.ones(nd)) / variance_parameter
    inverse_covariance = inverse_covariance.reshape(
        number_of_atoms, spatial_dimension, number_of_atoms, spatial_dimension
    )

    equilibrium_relative_coordinates = torch.from_numpy(
        get_silicon_supercell(supercell_factor=supercell_factor).astype(np.float32)
    )
    exact_samples = get_exact_samples(
        equilibrium_relative_coordinates, inverse_covariance, number_of_samples
    ).cpu()

    logger.info("Computing harmonic energies")
    exact_harmonic_energies = get_samples_harmonic_energy(
        equilibrium_relative_coordinates, inverse_covariance, exact_samples
    )

    logger.info("Extracting data artifacts")
    with open(data_pickle_path, "rb") as fd:
        recorded_data = torch.load(fd, map_location=torch.device("cpu"))

    energies = torch.load(energy_pickle_path)

    # ==============================================================
    logger.info("Plotting energy distributions")
    fig0 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig0.suptitle("Energy Distribution: Constrained vs Free")

    common_params = dict(density=True, bins=50, histtype="stepfilled", alpha=0.25)

    ax1 = fig0.add_subplot(111)
    ax1.hist(energies, **common_params, label="Repaint Samples Energies", color="red")
    ax1.hist(
        exact_harmonic_energies,
        **common_params,
        label="Unconstrained Samples Energies",
        color="green",
    )

    ax1.set_xlim(xmin=-0.01)
    ax1.set_xlabel("Unitless Harmonic Energy")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right", fancybox=True, shadow=True, ncol=1, fontsize=12)
    fig0.tight_layout()
    fig0.savefig(base_path / "comparing_free_and_repaint_energies.png")

    # ==============================================================

    sampling_times = recorded_data["time"]
    relative_coordinates = recorded_data["relative_coordinates"]
    batch_flat_relative_coordinates = einops.rearrange(
        relative_coordinates, "b t n d -> b t (n d)"
    )

    normalized_scores = recorded_data["normalized_scores"]
    batch_flat_normalized_scores = einops.rearrange(
        normalized_scores, "b t n d -> b t (n d)"
    )

    logger.info("Plotting scores along trajectories")
    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Root Mean Square Normalized Scores Along Sample Trajectories")
    rms_norm_score = (batch_flat_normalized_scores**2).mean(dim=-1).sqrt().numpy()

    constrained_rms_norm_score = (
        (batch_flat_normalized_scores[:, :, :number_of_constrained_coordinates] ** 2)
        .mean(dim=-1)
        .sqrt()
        .numpy()
    )

    free_rms_norm_score = (
        (batch_flat_normalized_scores[:, :, number_of_constrained_coordinates:] ** 2)
        .mean(dim=-1)
        .sqrt()
        .numpy()
    )

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    for ax, scores in zip(
        [ax1, ax2, ax3, ax4],
        [
            rms_norm_score,
            rms_norm_score,
            constrained_rms_norm_score,
            free_rms_norm_score,
        ],
    ):
        for y in scores[::10]:
            ax.plot(
                sampling_times, y, "-", color="gray", alpha=0.2, label="__nolabel__"
            )

    list_quantiles = [0.0, 0.10, 0.5, 1.0]
    list_colors = ["green", "yellow", "orange", "red"]

    for q, c in zip(list_quantiles, list_colors):
        energy_quantile = np.quantile(energies, q)
        idx = np.argmin(np.abs(energies - energy_quantile))
        e = energies[idx]
        for ax, scores in zip(
            [ax1, ax2, ax3, ax4],
            [
                rms_norm_score,
                rms_norm_score,
                constrained_rms_norm_score,
                free_rms_norm_score,
            ],
        ):
            ax.plot(
                sampling_times,
                scores[idx],
                "-",
                color=c,
                alpha=1.0,
                label=f"Q = {100 * q:2.0f}%, Energy: {e:5.1f}",
            )

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Diffusion Time")
        ax.set_ylabel(r"$\sqrt{\langle (\sigma(t) S_{\theta} )^2 \rangle}$")
        ax.set_xlim(1, 0)

    for ax in [ax2, ax3, ax4]:
        ax.set_yscale("log")

    ax1.set_title("All Coordinates (Not Log Scale)")
    ax1.set_ylim(ymin=0.0)

    ax2.set_title("All Coordinates")
    ax3.set_title("Restrained Coordinates")
    ax4.set_title("Free Coordinates")

    ax1.legend(loc=0, fontsize=6)
    fig.tight_layout()
    fig.savefig(base_path / "sampling_score_trajectories_repaint.png")
    plt.close(fig)

    logger.info("Done!")

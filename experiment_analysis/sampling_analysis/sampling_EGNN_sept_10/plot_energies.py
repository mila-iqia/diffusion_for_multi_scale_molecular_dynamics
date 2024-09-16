import glob
import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pymatgen.core import Structure

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.oracle.lammps import get_energy_and_forces_from_lammps
from crystal_diffusion.utils.logging_utils import setup_analysis_logger

plt.style.use(PLOT_STYLE_PATH)

logger = logging.getLogger(__name__)
setup_analysis_logger()

results_dir = Path(
    "/Users/bruno/courtois/partial_trajectory_sampling_EGNN_sept_10/partial_samples_EGNN_Sept_10"
)
reference_cif = results_dir / "reference_validation_structure.cif"


if __name__ == "__main__":
    times = np.linspace(0, 1, 1001)
    sigma_min = 0.001
    sigma_max = 0.5

    def sigma_function(times):
        """Compute sigma."""
        return sigma_min ** (1.0 - times) * sigma_max**times

    sigmas = sigma_function(times)

    special_times = [0.479, 0.668, 0.905]
    list_colors = ["green", "black", "red"]

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Noise Schedule")
    ax = fig.add_subplot(111)
    ax.plot(times, sigmas, "b-")

    for tf, c in zip(special_times, list_colors):
        sf = sigma_function(tf)
        ax.vlines(
            tf, 0, 0.5, colors=c, linestyles="dashed", label=r"$\sigma$ = " + f"{sf:5.4f}"
        )

    ax.set_xlabel(r"Time ($t_f$)")
    ax.set_ylabel(r"Noise $\sigma$")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 0.5])
    ax.legend(loc=0)
    plt.show()

    reference_structure = Structure.from_file(reference_cif)
    list_energy = []
    logger.info("Compute reference energy from Oracle")
    with tempfile.TemporaryDirectory() as tmp_work_dir:
        atom_types = np.array(len(reference_structure) * [1])
        positions = reference_structure.frac_coords @ reference_structure.lattice.matrix
        reference_energy, _ = get_energy_and_forces_from_lammps(
            positions,
            reference_structure.lattice.matrix,
            atom_types,
            tmp_work_dir=tmp_work_dir,
        )

    list_times = []
    list_energies = []
    for pickle_path in glob.glob(
        str(results_dir / "diffusion_energies_sample_time=*.pt")
    ):
        energies = torch.load(pickle_path).numpy()
        time = float(pickle_path.split("=")[1].split(".pt")[0])

        list_times.append(time)
        list_energies.append(energies)

    times = np.array(list_times)
    energies = np.array(list_energies)

    sorting_indices = np.argsort(times)
    times = times[sorting_indices]
    energies = energies[sorting_indices]

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Energy Quantiles for Partial Trajectories Final Point")
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    list_axes = [ax1, ax2, ax3, ax4]
    list_q = np.linspace(0, 1, 21)
    number_of_times = len(times)

    for ax, indices in zip(
        list_axes, np.split(np.arange(number_of_times), len(list_axes))
    ):
        # Ad Hoc hack so we can see something
        e_min = np.min([energies[indices].min(), reference_energy])
        e_95 = np.quantile(energies[indices], 0.95)
        delta = (e_95 - e_min) / 10.0
        e_max = e_95 + delta

        ax.set_ylim(e_min - 0.1, e_max)

        ax.hlines(
            reference_energy,
            0,
            100,
            color="black",
            linestyles="dashed",
            label="Reference Energy",
        )

        for idx in indices:
            tf = times[idx]
            time_energies = energies[idx]
            energy_quantiles = np.quantile(time_energies, list_q)
            ax.plot(100 * list_q, energy_quantiles, "-", label=f"time = {tf:3.2f}")

        ax.legend(loc=0, fontsize=7)
        ax.set_xlim([-0.1, 100.1])
        ax.set_xlabel("Quantile (%)")
        ax.set_ylabel("Energy (eV)")
    fig.tight_layout()
    plt.show()

    fig2 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig2.suptitle("Energy Extrema of Trajectory End Points")
    ax = fig2.add_subplot(111)
    ax.plot(times, energies.min(axis=1), "o-", label="Minimum Energy")
    ax.plot(
        times, np.quantile(energies, 0.5, axis=1), "o-", label="50% Quantile Energy"
    )
    ax.plot(times, energies.max(axis=1), "o-", label="Maximum Energy")
    ax.set_xlabel("Starting Diffusion Time, $t_f$")
    ax.set_ylabel("Energy (eV)")

    ax.hlines(
        reference_energy,
        0,
        100,
        color="black",
        linestyles="dashed",
        label="Reference Energy",
    )

    ax.set_ylim(-280, -120)
    ax.set_xlim(0.0, 1.01)
    ax.legend(loc=0)

    plt.show()

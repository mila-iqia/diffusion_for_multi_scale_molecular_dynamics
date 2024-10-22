"""Diffusion Dataset analysis.

This script computes and plots different features of a dataset used to train a diffusion model.
"""

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffusion_for_multi_scale_molecular_dynamics import (ANALYSIS_RESULTS_DIR,
                                                          DATA_DIR)
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.data.parse_lammps_outputs import \
    parse_lammps_output

DATASET_NAME = "si_diffusion_v1"


def read_lammps_run(run_path: str) -> pd.DataFrame:
    """Read and organize the LAMMPS output files in a dataframe.

    Args:
        run_path: path to LAMMPS output directory. Should contain a dump file and a thermo log file.

    Returns:
        output as
    """
    dump_file = [d for d in os.listdir(run_path) if "dump" in d]
    thermo_file = [d for d in os.listdir(run_path) if "thermo" in d]

    df = parse_lammps_output(
        os.path.join(run_path, dump_file[0]),
        os.path.join(run_path, thermo_file[0]),
        None,
    )

    return df


def compute_metrics_for_a_run(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Get the energy, forces average, RMS displacement and std dev for a single MD run.

    Args:
        df: LAMMPS output organized in a DataFrame.

    Returns:
        metrics evaluated at each MD step organized in a dict
    """
    metrics = {}
    metrics["energy"] = df["energy"]
    force_norm_mean = df.apply(
        lambda row: np.mean(
            [
                np.sqrt(fx**2 + fy**2 + fz**2)
                for fx, fy, fz in zip(row["fx"], row["fy"], row["fz"])
            ]
        ),
        axis=1,
    )
    metrics["force_norm_average"] = force_norm_mean

    x0s = df["x"][0]
    y0s = df["y"][0]
    z0s = df["z"][0]

    square_displacement = df.apply(
        lambda row: [
            (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
            for x, y, z, x0, y0, z0 in zip(row["x"], row["y"], row["z"], x0s, y0s, z0s)
        ],
        axis=1,
    )

    metrics["root_mean_square_displacement"] = square_displacement.apply(
        lambda row: np.sqrt(np.mean(row))
    )

    metrics["std_displacement"] = np.std(square_displacement.apply(np.sqrt))

    return metrics


def plot_metrics_runs(dataset_name: str, mode: str = "train"):
    """Compute and plot metrics for a dataset made up of several MD runs.

    Args:
        dataset_name: name of the dataset - should match the name of the folder in DATA_DIR
        mode (optional): analyze train or valid data. Defaults to train.
    """
    assert mode in ["train", "valid"], f"Mode should be train or valid. Got {mode}"
    dataset_path = os.path.join(DATA_DIR, dataset_name)

    list_runs = [
        d
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
        and d.startswith(f"{mode}_run")
        and not d.endswith("backup")
    ]

    metrics = {}
    for run in list_runs:
        df = read_lammps_run(os.path.join(dataset_path, run))
        metrics_run = compute_metrics_for_a_run(df)
        metrics[run] = metrics_run

    plt.style.use(PLOT_STYLE_PATH)

    fig, axs = plt.subplots(
        4, 1, figsize=(PLEASANT_FIG_SIZE[0], 4 * PLEASANT_FIG_SIZE[1])
    )
    fig.suptitle("MD runs properties")

    # energy
    axs[0].set_title("Energy")
    axs[0].set_ylabel("Energy (kcal / mol)")
    # forces
    axs[1].set_title("Force Norm Averaged over Atoms")
    axs[1].set_ylabel(r"Force Norm (g/mol * Angstrom / fs^2)")
    # mean squared displacement
    axs[2].set_title("RMS Displacement")
    axs[2].set_ylabel("RMSD (Angstrom)")
    # std squared displacement
    axs[3].set_title("Std-Dev Displacement")
    axs[3].set_ylabel("Std Displacement (Angstrom)")

    legend = []
    for k, m in metrics.items():
        axs[0].plot(m["energy"], "-", lw=2)
        axs[1].plot(m["force_norm_average"], ":", lw=2)
        axs[2].plot(m["root_mean_square_displacement"], lw=2)
        axs[3].plot(m["std_displacement"], lw=2)
        legend.append(k)

    for ax in axs:
        ax.legend(legend)
        ax.set_xlabel("MD step")

    fig.tight_layout()

    fig.savefig(
        ANALYSIS_RESULTS_DIR.joinpath(f"{dataset_name}_{mode}_analysis.png"), dpi=300
    )


def main():
    """Analyze training and validation set of a dataset."""
    plot_metrics_runs(DATASET_NAME, mode="train")
    plot_metrics_runs(DATASET_NAME, mode="valid")


if __name__ == "__main__":
    main()

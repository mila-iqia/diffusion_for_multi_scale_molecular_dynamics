import glob
from collections import defaultdict
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.namespace import \
    AXL_COMPOSITION
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import \
    VarianceScheduler
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import \
    get_periodic_adjacency_information

plt.style.use(PLOT_STYLE_PATH)


def get_number_of_samples_with_overlaps(samples_pickle, radial_cutoff=1.0):
    """Get number of samples with overlaps."""
    data = torch.load(samples_pickle, map_location="cpu")
    compositions = data[AXL_COMPOSITION]
    relative_coordinates = compositions.X
    basis_vectors = compositions.L
    cartesian_positions = einops.einsum(
        relative_coordinates,
        basis_vectors,
        "batch natoms space1, batch space1 space -> batch natoms space",
    )
    adjacency_info = get_periodic_adjacency_information(
        cartesian_positions, basis_vectors, radial_cutoff=radial_cutoff
    )

    edge_batch_indices = adjacency_info.edge_batch_indices

    number_of_short_edges = len(edge_batch_indices) // 2
    number_of_samples_with_overlaps = len(edge_batch_indices.unique())
    return number_of_samples_with_overlaps, number_of_short_edges


schedule_type = "exponential"
# schedule_type = 'linear'

samples_top_dir = Path(
    "/Users/brunorousseau/courtois/jan11_egnn_sige_2x2x2/sampling_constraints/output"
)

noise_parameters = NoiseParameters(
    total_time_steps=1000, sigma_min=0.0001, sigma_max=0.2, schedule_type=schedule_type
)

if __name__ == "__main__":

    if schedule_type == "linear":
        output_directory_template = "output_T={}"
    elif schedule_type == "exponential":
        output_directory_template = "output_exponential_T={}"

    list_file_starting_time = []
    for path in glob.glob(str(samples_top_dir / output_directory_template.format("*"))):
        starting_time = int(path.split("=")[1])
        list_file_starting_time.append(starting_time)

    list_file_starting_time = np.sort(list_file_starting_time)

    overlap_counts = defaultdict(list)
    short_edge_counts = defaultdict(list)

    list_radial_cutoffs = [1.0, 2.0]

    list_starting_time = []
    for start_time in tqdm(list_file_starting_time):
        samples_pickle = (
            samples_top_dir
            / output_directory_template.format(start_time)
            / "samples.pt"
        )
        if not samples_pickle.is_file():
            continue
        list_starting_time.append(start_time)
        for radial_cutoff in list_radial_cutoffs:
            noverlaps, number_of_short_edges = get_number_of_samples_with_overlaps(
                samples_pickle, radial_cutoff=radial_cutoff
            )
            overlap_counts[radial_cutoff].append(noverlaps)
            short_edge_counts[radial_cutoff].append(number_of_short_edges)

    variance_scheduler = VarianceScheduler(noise_parameters)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(
        f"Counting Short Edges in 128 Drawn Samples"
        f"\n T=1000, {schedule_type} Schedule, Adapator Corrector."
    )
    ax1 = fig.add_subplot(111)

    for radial_cutoff in list_radial_cutoffs:
        counts = short_edge_counts[radial_cutoff]

        ax1.plot(
            list_starting_time,
            counts,
            "o-",
            ms=5,
            mew=1,
            label=r"Radial Cutoff: {} $\AA$".format(radial_cutoff),
        )

    ymax = max(counts) + 100

    ax1.set_xlabel("Free Diffusion Starting Time Index")
    ax1.set_ylabel("Number of Short Edges")
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)

    ax1.set_ylim(ymin=-1, ymax=ymax)
    ax1.set_xlim(0, 1000)

    ax1.legend(loc=2)

    list_times = torch.linspace(0.0, 1.0, 1000)
    sigmas = variance_scheduler.get_sigma(list_times)

    ax2 = ax1.twinx()
    if schedule_type == "linear":
        ax2.vlines(
            165,
            0,
            ymax,
            linestyles="--",
            colors="green",
            alpha=0.5,
            label="__nolabel__",
        )
        ax2.hlines(
            0.0331,
            0,
            1000,
            linestyles="--",
            colors="green",
            alpha=0.5,
            label=r"$\sigma$ = 0.0331",
        )
        ax2.legend(loc=0)
    if schedule_type == "exponential":
        ax2.vlines(
            745,
            0,
            ymax,
            linestyles="--",
            colors="green",
            alpha=0.5,
            label="__nolabel__",
        )
        ax2.hlines(
            0.0288,
            0,
            1000,
            linestyles="--",
            colors="green",
            alpha=0.5,
            label=r"$\sigma$ = 0.0288",
        )
        ax2.legend(loc=1)

    ax2.plot(1000 * list_times, sigmas, "b-")
    ax2.set_ylabel(r"$\sigma(t)$", color="blue")
    ax2.set_ylim(0, 0.2)

    fig.tight_layout()
    plt.show()

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
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_lattice_parameters_to_unit_cell_vectors
from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import \
    get_periodic_adjacency_information

plt.style.use(PLOT_STYLE_PATH)


def get_number_of_samples_with_overlaps(samples_pickle, radial_cutoff=1.0):
    """Get number of samples with overlaps."""
    data = torch.load(samples_pickle, map_location="cpu")
    compositions = data[AXL_COMPOSITION]
    relative_coordinates = compositions.X

    lattice_parameters = compositions.L
    basis_vectors = map_lattice_parameters_to_unit_cell_vectors(lattice_parameters)

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


image_directory = Path(__file__).parent / "images"
image_directory.mkdir(exist_ok=True)

schedule_type = "exponential"
# schedule_type = 'linear'

# workdir = "july26_si_egnn_2x2x2"
# system = "Si 2x2x2"

workdir = "july26_si_egnn_3x3x3"
system = "Si_3x3x3"

base_experiment_dir = Path("/Users/brunorousseau/courtois/july26/")

samples_top_dir = base_experiment_dir / workdir / "constrained_samples_from_run1/output" / schedule_type

noise_parameters = NoiseParameters(total_time_steps=1000, sigma_min=0.0001, sigma_max=0.2, schedule_type=schedule_type)

if __name__ == "__main__":

    list_file_starting_time = []
    for path in glob.glob(str(samples_top_dir / "output_T=*")):
        starting_time = int(path.split("=")[1])
        list_file_starting_time.append(starting_time)

    list_file_starting_time = np.sort(list_file_starting_time)

    overlap_counts = defaultdict(list)
    short_edge_counts = defaultdict(list)

    list_radial_cutoffs = [1.0, 2.0]

    list_starting_time = []
    for start_time in tqdm(list_file_starting_time):
        output_directory = f"output_T={start_time}"
        samples_pickle = samples_top_dir / output_directory / "samples.pt"

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
    system_for_title = system.replace('_', ' ').capitalize()
    fig.suptitle(f"Counting Short Edges in {system_for_title} Samples\n{schedule_type.capitalize()} Noise Schedule")
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

    ax1.set_ylim(ymin=-1)
    ax1.set_xlim(0, 1000)

    ax1.legend(loc=2)

    list_times = torch.linspace(0.0, 1.0, 1000)
    sigmas = variance_scheduler.get_sigma(list_times)

    ax2 = ax1.twinx()

    ax2.plot(1000 * list_times, sigmas, "b-")
    ax2.set_ylabel(r"$\sigma(t)$", color="blue")
    ax1.set_ylim(ymin=0.0)
    ax2.set_ylim(0.0, 0.2)

    fig.tight_layout()
    output_file = image_directory / f"short_edges_vs_start_times_{schedule_type}_{system}.png"
    fig.savefig(output_file)
    plt.close(fig)

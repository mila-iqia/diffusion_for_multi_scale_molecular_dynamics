import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops import einops

from crystal_diffusion import DATA_DIR
from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.data.diffusion.data_loader import (
    LammpsForDiffusionDataModule, LammpsLoaderParameters)
from crystal_diffusion.models.mace_utils import get_adj_matrix
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from experiment_analysis import EXPERIMENT_ANALYSIS_DIR


def get_interatomic_distances(cartesian_positions: torch.Tensor,
                              basis_vectors: torch.Tensor,
                              radial_cutoff: float = 5.0):
    """Get interatomic distances."""
    shifted_adjacency_matrix, shifts, batch_indices = get_adj_matrix(positions=cartesian_positions,
                                                                     basis_vectors=basis_vectors,
                                                                     radial_cutoff=radial_cutoff)

    flat_positions = einops.rearrange(cartesian_positions, "b n d -> (b n) d")

    displacements = flat_positions[shifted_adjacency_matrix[1]] - flat_positions[shifted_adjacency_matrix[0]] + shifts
    interatomic_distances = torch.linalg.norm(displacements, dim=1)
    return interatomic_distances


logger = logging.getLogger(__name__)

plt.style.use(PLOT_STYLE_PATH)


base_data_dir = Path("/Users/bruno/courtois/difface_ode/run1")
position_samples_dir = base_data_dir / "diffusion_position_samples"
energy_samples_dir = base_data_dir / "energy_samples"


dataset_name = "si_diffusion_1x1x1"
lammps_run_dir = str(DATA_DIR / dataset_name)
processed_dataset_dir = str(DATA_DIR / dataset_name / 'processed')
data_params = LammpsLoaderParameters(batch_size=64, max_atom=8)
cache_dir = str(EXPERIMENT_ANALYSIS_DIR / "cache" / dataset_name)

epoch = 5


if __name__ == '__main__':

    datamodule = LammpsForDiffusionDataModule(
        lammps_run_dir=lammps_run_dir,
        processed_dataset_dir=processed_dataset_dir,
        hyper_params=data_params,
        working_cache_dir=cache_dir,
    )

    datamodule.setup()

    train_dataset = datamodule.train_dataset
    batch = train_dataset[:1000]

    positions_data = torch.load(position_samples_dir / f"diffusion_position_sample_epoch={epoch}_steps=0.pt",
                                map_location=torch.device('cpu'))

    unit_cell = positions_data['unit_cell']

    batch_times = positions_data['time'][0]
    batch_noisy_relative_coordinates = positions_data['relative_coordinates'][0]
    number_of_atoms, spatial_dimension = batch_noisy_relative_coordinates.shape[-2:]

    batch_flat_noisy_relative_coordinates = einops.rearrange(batch_noisy_relative_coordinates,
                                                             "b t n d -> b t (n d)")

    fig1 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig1.suptitle(f'ODE Trajectories: Sample at Epoch {epoch}')

    ax = fig1.add_subplot(111)
    ax.set_xlabel('Diffusion Time')
    ax.set_ylabel('Raw Relative Coordinate')
    ax.yaxis.tick_right()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    time = batch_times[0]  # all time arrays are the same
    for flat_relative_coordinates in batch_flat_noisy_relative_coordinates:
        for i in range(number_of_atoms * spatial_dimension):
            coordinate = flat_relative_coordinates[:, i]
            ax.plot(time.cpu(), coordinate.cpu(), '-', color='b', alpha=0.05)

    ax.set_xlim([1.01, -0.01])
    ax.set_ylim([-2.0, 2.0])
    plt.show()

    training_relative_coordinates = batch['relative_coordinates']
    training_center_of_mass = training_relative_coordinates.mean(dim=1).mean(dim=0)

    raw_sample_relative_coordinates = map_relative_coordinates_to_unit_cell(batch_noisy_relative_coordinates[:, -1])
    raw_sample_centers_of_mass = raw_sample_relative_coordinates.mean(dim=1)

    zero_centered_sample_relative_coordinates = (raw_sample_relative_coordinates
                                                 - raw_sample_centers_of_mass.unsqueeze(1))
    sample_relative_coordinates = (zero_centered_sample_relative_coordinates
                                   + training_center_of_mass.unsqueeze(0).unsqueeze(0))

    sample_relative_coordinates = map_relative_coordinates_to_unit_cell(sample_relative_coordinates)

    fig2 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig2.suptitle(f'ODE Marginal Distributions, Sample  at Epoch {epoch}')
    ax1 = fig2.add_subplot(131, aspect='equal')
    ax2 = fig2.add_subplot(132, aspect='equal')
    ax3 = fig2.add_subplot(133, aspect='equal')

    xs = einops.rearrange(sample_relative_coordinates, 'b n d -> (b n) d')
    zs = einops.rearrange(training_relative_coordinates, 'b n d -> (b n) d')
    ax1.set_title('XY Projection')
    ax1.plot(xs[:, 0], xs[:, 1], 'ro', alpha=0.5, mew=0, label='ODE Solver Samples')
    ax1.plot(zs[:, 0], zs[:, 1], 'go', alpha=0.05, mew=0, label='Training Data')

    ax2.set_title('XZ Projection')
    ax2.plot(xs[:, 0], xs[:, 2], 'ro', alpha=0.5, mew=0, label='ODE Solver Samples')
    ax2.plot(zs[:, 0], zs[:, 2], 'go', alpha=0.05, mew=0, label='Training Data')

    ax3.set_title('YZ Projection')
    ax3.plot(xs[:, 1], xs[:, 2], 'ro', alpha=0.5, mew=0, label='ODE Solver Samples')
    ax3.plot(zs[:, 1], zs[:, 2], 'go', alpha=0.05, mew=0, label='Training Data')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.vlines(x=[0, 1], ymin=0, ymax=1, color='k', lw=2)
        ax.hlines(y=[0, 1], xmin=0, xmax=1, color='k', lw=2)

    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2, fancybox=True, shadow=True)
    fig2.tight_layout()
    plt.show()

    fig3 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    ax1 = fig3.add_subplot(131)
    ax2 = fig3.add_subplot(132)
    ax3 = fig3.add_subplot(133)
    fig3.suptitle(f"Marginal Distributions of t=0 Samples, Sample at Epoch {epoch}")

    common_params = dict(histtype='stepfilled', alpha=0.5, bins=50)

    ax1.hist(xs[:, 0], **common_params, facecolor='r', label='ODE solver')
    ax2.hist(xs[:, 1], **common_params, facecolor='r', label='ODE solver')
    ax3.hist(xs[:, 2], **common_params, facecolor='r', label='ODE solver')

    ax1.hist(zs[:, 0], **common_params, facecolor='g', label='Training Data')
    ax2.hist(zs[:, 1], **common_params, facecolor='g', label='Training Data')
    ax3.hist(zs[:, 2], **common_params, facecolor='g', label='Training Data')

    ax1.set_xlabel('X')
    ax2.set_xlabel('Y')
    ax3.set_xlabel('Z')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-0.01, 1.01)
        ax.set_yscale('log')

    ax1.legend(loc=0)
    fig3.tight_layout()
    plt.show()

    radial_cutoff = 5.4
    training_cartesian_positions = batch['cartesian_positions']
    basis_vectors = torch.diag_embed(batch['box'])
    training_interatomic_distances = get_interatomic_distances(training_cartesian_positions,
                                                               basis_vectors,
                                                               radial_cutoff=radial_cutoff)

    sample_relative_coordinates = map_relative_coordinates_to_unit_cell(batch_noisy_relative_coordinates[:, -1])
    sample_cartesian_positions = torch.bmm(sample_relative_coordinates, unit_cell)
    sample_interatomic_distances = get_interatomic_distances(sample_cartesian_positions,
                                                             unit_cell,
                                                             radial_cutoff=radial_cutoff)

    fig4 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig4.suptitle(f'Interatomic Distance Distribution: Sample at Epoch {epoch}')

    ax1 = fig4.add_subplot(121)
    ax2 = fig4.add_subplot(122)

    ax1.set_title('Training vs. Samples')
    ax2.set_title('Intermediate Diffusion')

    common_params = dict(histtype='stepfilled', alpha=0.5, bins=75)
    ax1.hist(training_interatomic_distances, **common_params, facecolor='g', label='Training Data')
    ax1.hist(sample_interatomic_distances, **common_params, facecolor='r', label='ODE Sample, t = 0')

    for time_idx, color in zip([0, len(time) // 2 + 1, -1], ['blue', 'yellow', 'red']):
        sample_relative_coordinates = map_relative_coordinates_to_unit_cell(
            batch_noisy_relative_coordinates[:, time_idx])
        sample_cartesian_positions = torch.bmm(sample_relative_coordinates, unit_cell)
        sample_interatomic_distances = get_interatomic_distances(sample_cartesian_positions,
                                                                 unit_cell,
                                                                 radial_cutoff=radial_cutoff)
        ax2.hist(sample_interatomic_distances, **common_params, facecolor=color,
                 label=f'Noisy Sample t = {time[time_idx]:2.1f}')

    for ax in [ax1, ax2]:
        ax.set_xlabel('Distance (Angstrom)')
        ax.set_ylabel('Count')
        ax.set_xlim([-0.01, radial_cutoff])
        ax.legend(loc=0)
        ax.set_yscale('log')
    fig4.tight_layout()
    plt.show()

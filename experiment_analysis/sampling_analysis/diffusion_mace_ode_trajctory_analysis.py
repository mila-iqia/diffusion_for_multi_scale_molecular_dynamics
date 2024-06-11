import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH

logger = logging.getLogger(__name__)

plt.style.use(PLOT_STYLE_PATH)


# Some hardcoded paths and parameters. Change as needed!
epoch = 30
base_data_dir = Path("/Users/bruno/courtois/difface_ode/run7")
position_samples_dir = base_data_dir / "diffusion_position_samples"
energy_samples_dir = base_data_dir / "energy_samples"
energy_data_directory = base_data_dir / "energy_samples"


if __name__ == '__main__':
    energies = torch.load(energy_data_directory / f"energies_sample_epoch={epoch}.pt")
    positions_data = torch.load(position_samples_dir / f"diffusion_position_sample_epoch={epoch}_steps=0.pt",
                                map_location=torch.device('cpu'))

    unit_cell = positions_data['unit_cell']

    batch_times = positions_data['time'][0]
    batch_noisy_relative_coordinates = positions_data['relative_coordinates'][0]
    number_of_atoms, spatial_dimension = batch_noisy_relative_coordinates.shape[-2:]

    idx = energies.argmax()
    relative_coordinates = batch_noisy_relative_coordinates[idx]

    fig1 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig1.suptitle(f'ODE trajectory: Sample {idx} at Epoch {epoch} - Energy = {energies.max():4.2f}')

    ax1 = fig1.add_subplot(131)
    ax2 = fig1.add_subplot(132)
    ax3 = fig1.add_subplot(133)

    time = batch_times[0]  # all time arrays are the same

    for atom_idx in range(number_of_atoms):
        ax1.plot(time, relative_coordinates[:, atom_idx, 0], '-', alpha=0.5)
        ax2.plot(time, relative_coordinates[:, atom_idx, 1], '-', alpha=0.5)
        ax3.plot(time, relative_coordinates[:, atom_idx, 2], '-', alpha=0.5)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Diffusion Time')
        ax.set_ylabel('Raw Relative Coordinate')
        ax.yaxis.tick_right()
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.set_xlim([1.01, -0.01])
    ax1.set_ylabel('X')
    ax2.set_ylabel('Y')
    ax3.set_ylabel('Z')
    fig1.tight_layout()
    plt.show()

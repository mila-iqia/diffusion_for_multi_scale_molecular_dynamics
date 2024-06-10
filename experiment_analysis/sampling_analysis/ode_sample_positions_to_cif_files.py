"""Position to cif files for the ODE sampler.

A simple script to extract the diffusion positions from a pickle on disk and output
in cif format for visualization.
"""
from pathlib import Path

import torch
from pymatgen.core import Lattice, Structure

from crystal_diffusion.utils.sample_trajectory import ODESampleTrajectory

# Hard coding some paths to local results. Modify as needed...
epoch = 15

base_data_dir = Path("/Users/bruno/courtois/difface_ode/run1")
trajectory_data_directory = base_data_dir / "diffusion_position_samples"
energy_data_directory = base_data_dir / "energy_samples"
output_top_dir = trajectory_data_directory.parent / "visualization"


if __name__ == '__main__':
    energies = torch.load(energy_data_directory / f"energies_sample_epoch={epoch}.pt")

    sample_idx = energies.argmax()
    output_dir = output_top_dir / f"visualise_sampling_trajectory_epoch_{epoch}_sample_{sample_idx}"
    output_dir.mkdir(exist_ok=True, parents=True)

    pickle_path = trajectory_data_directory / f"diffusion_position_sample_epoch={epoch}_steps=0.pt"
    sample_trajectory = ODESampleTrajectory.read_from_pickle(pickle_path)

    basis_vectors = sample_trajectory.data['unit_cell'][sample_idx].numpy()
    lattice = Lattice(matrix=basis_vectors, pbc=(True, True, True))

    # Shape [batch, time, number of atoms, space dimension]
    batch_noisy_relative_coordinates = sample_trajectory.data['relative_coordinates'][0]

    noisy_relative_coordinates = batch_noisy_relative_coordinates[sample_idx].numpy()

    for idx, coordinates in enumerate(noisy_relative_coordinates):
        number_of_atoms = coordinates.shape[0]
        species = number_of_atoms * ['Si']

        structure = Structure(lattice=lattice,
                              species=species,
                              coords=coordinates,
                              coords_are_cartesian=False)

        file_path = str(output_dir / f"diffusion_positions_{idx}.cif")
        structure.to_file(file_path)

"""Plotting Repulsive Forces.

This script plots the repulsive force field that can be used to prevent atoms overlapping at sampling time.
"""

from pathlib import Path

import torch
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.force_field_augmented_score_network import (
    ForceFieldAugmentedScoreNetwork, ForceFieldParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISY_AXL_COMPOSITION, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_unit_cell_to_lattice_parameters

PLOTS_OUTPUT_DIRECTORY = Path(__file__).parent / "images"
PLOTS_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

plt.style.use(PLOT_STYLE_PATH)

natoms = 2
spatial_dimension = 3
batch_size = 1


radial_cutoff = 1.5
strength = 20

acell = 10.86

force_field_parameters = ForceFieldParameters(
    radial_cutoff=radial_cutoff, strength=strength
)

if __name__ == "__main__":
    # The score network will not be called since we are only looking to plot the force field.
    # This is why we can get away with giving a None argument.
    force_field_score_network = ForceFieldAugmentedScoreNetwork(
        score_network=None, force_field_parameters=force_field_parameters
    )

    list_x = torch.linspace(0.0, 1.0, 1001)
    list_f = []

    for x in list_x:
        relative_coordinates = torch.tensor(
            [[[0.5 - 0.5 * x, 0.5, 0.0], [0.5 + 0.5 * x, 0.5, 0.0]]]
        )
        # The atom types are irrelevant to the force field. We put a dummy value.
        batch_size, natoms, _ = relative_coordinates.shape
        atom_types = torch.zeros(batch_size, natoms)

        basis_vectors = acell * torch.diag(torch.ones(spatial_dimension)).unsqueeze(0)
        lattice_parameters = map_unit_cell_to_lattice_parameters(basis_vectors)
        batch = {
            NOISY_AXL_COMPOSITION: AXL(A=atom_types, X=relative_coordinates, L=lattice_parameters),
            UNIT_CELL: basis_vectors,
        }
        forces = force_field_score_network.get_relative_coordinates_pseudo_force(batch)
        f = torch.abs(forces[0, 0, 0])
        list_f.append(f)

    list_f = torch.tensor(list_f)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(
        f"Repulsive Pseudo Force for Strength = {strength} and radial cutoff = {radial_cutoff} $\\AA$"
    )
    ax = fig.add_subplot(111)

    ax.plot(acell * list_x, list_f, "b-")
    ax.set_xlim(0, 4)
    ax.set_ylim(ymin=-0.1)
    ax.set_xlabel(r"Interatomic Distance ($\AA$)")
    ax.set_ylabel("Magnitude of Pseudo Force")
    fig.tight_layout()
    fig.savefig(PLOTS_OUTPUT_DIRECTORY / "repulsive_force.png")
    plt.show()

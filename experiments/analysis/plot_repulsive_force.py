import torch
from crystal_diffusion.models.score_networks.force_field_augmented_score_network import (
    ForceFieldAugmentedScoreNetwork, ForceFieldParameters)
from crystal_diffusion.namespace import NOISY_RELATIVE_COORDINATES, UNIT_CELL
from matplotlib import pyplot as plt
from src.crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH

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
    force_field_score_network = ForceFieldAugmentedScoreNetwork(
        score_network=None, force_field_parameters=force_field_parameters
    )

    list_x = torch.linspace(0.0, 1.0, 1001)
    list_f = []

    for x in list_x:
        relative_coordinates = torch.tensor(
            [[[0.5 - 0.5 * x, 0.5, 0.0], [0.5 + 0.5 * x, 0.5, 0.0]]]
        )

        basis_vectors = acell * torch.diag(torch.ones(spatial_dimension)).unsqueeze(0)
        batch = {
            NOISY_RELATIVE_COORDINATES: relative_coordinates,
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
    plt.show()

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.trajectory_initializer import (
    FullRandomTrajectoryInitializer, TrajectoryInitializerParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_unit_cell_to_lattice_parameters
from diffusion_for_multi_scale_molecular_dynamics.utils.reference_configurations import \
    create_equilibrium_sige_structure


class FixedStartingPointTrajectoryInitializer(FullRandomTrajectoryInitializer):
    """Fixed starting point trajectory initializer."""

    def __init__(self):
        """Init method."""
        structure = create_equilibrium_sige_structure()
        self.fixed_relative_coordinates = torch.from_numpy(structure.frac_coords).to(
            torch.float
        )
        basis_vectors = torch.from_numpy(1.0 * structure.lattice.matrix).to(torch.float)
        fixed_lattice_parameters = map_unit_cell_to_lattice_parameters(basis_vectors)

        trajectory_initializer_parameters = TrajectoryInitializerParameters(
            spatial_dimension=3,
            num_atom_types=2,
            use_fixed_lattice_parameters=True,
            fixed_lattice_parameters=fixed_lattice_parameters,
            number_of_atoms=8,
        )

        super().__init__(trajectory_initializer_parameters)

    def initialize(self, number_of_samples: int, device: torch.device) -> AXL:
        """Initialize."""
        random_init_composition = super().initialize(number_of_samples, device)

        fixed_relative_coordinates = einops.repeat(
            self.fixed_relative_coordinates, "n d -> b n d", b=number_of_samples
        )

        init_composition = AXL(
            A=random_init_composition.A,
            X=fixed_relative_coordinates,
            L=random_init_composition.L,
        )

        return init_composition


if __name__ == "__main__":

    # Sanity check that we get good initial compositions.
    trajectory_initializer = FixedStartingPointTrajectoryInitializer()

    number_of_samples = 16

    initial_composition = trajectory_initializer.initialize(
        number_of_samples=number_of_samples, device=torch.device("cpu")
    )

    relative_coordinates = torch.from_numpy(
        create_equilibrium_sige_structure().frac_coords
    ).to(torch.float)

    expected_x = einops.repeat(
        relative_coordinates, "n d -> b n d", b=number_of_samples
    )

    torch.testing.assert_allclose(initial_composition.X, expected_x)

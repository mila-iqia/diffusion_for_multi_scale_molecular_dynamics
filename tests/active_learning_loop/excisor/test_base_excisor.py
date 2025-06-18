import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.no_op_excisor import (
    NoOpExcision, NoOpExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_number_of_lattice_parameters


class TestBaseEnvironmentExcision:

    @pytest.fixture()
    def excisor(self):
        return NoOpExcision(NoOpExcisionArguments())

    @pytest.fixture
    def number_of_atoms(self):
        return 20

    @pytest.fixture
    def uncertainty_per_atom(self, number_of_atoms):
        return np.random.random(number_of_atoms)

    @pytest.fixture(params=[1, 2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture
    def num_atom_types(self):
        return 6

    @pytest.fixture
    def uncentered_axl_structure(
        self, number_of_atoms, num_atom_types, spatial_dimension
    ):
        rnd_type = np.random.randint(
            low=0, high=num_atom_types, size=(number_of_atoms,)
        )
        rnd_coordinates = np.random.rand(number_of_atoms, spatial_dimension)
        rnd_lattice = np.random.rand(
            get_number_of_lattice_parameters(spatial_dimension)
        )
        uncentered_axl = AXL(A=rnd_type, X=rnd_coordinates, L=rnd_lattice)
        return uncentered_axl

    def calculated_centered_coordinates(self, center_idx, uncentered_coordinates):
        central_atom_coord = uncentered_coordinates[center_idx, :]
        new_atom_positions = []
        box_center = np.array([0.5] * uncentered_coordinates.shape[-1])
        for atom_idx in range(uncentered_coordinates.shape[0]):
            shifted_coordinates = [
                (x + bc - ca) % 1
                for x, bc, ca in zip(
                    uncentered_coordinates[atom_idx], box_center, central_atom_coord
                )
            ]
            new_atom_positions.append(shifted_coordinates)
        return np.stack(new_atom_positions)

    def test_center_structure(self, number_of_atoms, uncentered_axl_structure, excisor):
        for central_atom_idx in range(number_of_atoms):
            expected_shifted_coordinates = self.calculated_centered_coordinates(
                central_atom_idx, uncentered_axl_structure.X
            )
            calculated_shifted_structure = excisor.center_structure(
                uncentered_axl_structure, central_atom_idx
            )
            assert np.array_equal(
                uncentered_axl_structure.A, calculated_shifted_structure.A
            )
            assert np.array_equal(
                uncentered_axl_structure.L, calculated_shifted_structure.L
            )
            assert np.allclose(
                expected_shifted_coordinates, calculated_shifted_structure.X
            )

import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from tests.active_learning_loop.conftest import BaseTestAxlStructure


class BaseTestExcision(BaseTestAxlStructure):

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        np.random.seed(345343)

    @pytest.fixture
    def excisor(self, **kwargs):
        raise NotImplementedError("Must be implemented in subclass")

    @pytest.fixture
    def expected_excised_environment(self, **kwargs):
        raise NotImplementedError("must be implemented in subclass")

    @pytest.fixture
    def expected_excised_atom_index(self, **kwargs):
        raise NotImplementedError("must be implemented in subclass")

    def test_excise_one_environment(
        self,
        excisor,
        structure_axl,
        central_atom_idx,
        expected_excised_environment,
        expected_excised_atom_index
    ):
        calculated_environment, calculated_excised_central_atom_index = excisor._excise_one_environment(
            structure_axl, central_atom_idx
        )

        assert calculated_excised_central_atom_index == expected_excised_atom_index
        assert np.array_equal(expected_excised_environment.A, calculated_environment.A)
        assert np.array_equal(expected_excised_environment.X, calculated_environment.X)

    @pytest.fixture
    def expected_centered_axl_structure(self, central_atom_idx, structure_axl):
        uncentered_coordinates = structure_axl.X
        central_atom_coord = uncentered_coordinates[central_atom_idx, :]
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

        centered_axl_structure = AXL(A=structure_axl.A,
                                     X=np.stack(new_atom_positions),
                                     L=structure_axl.L)

        return centered_axl_structure

    def test_center_structure(self, central_atom_idx, expected_centered_axl_structure, structure_axl, excisor):
        calculated_shifted_structure = excisor.center_structure(
            structure_axl, central_atom_idx
        )
        assert np.array_equal(
            expected_centered_axl_structure.A, calculated_shifted_structure.A
        )
        assert np.array_equal(
            expected_centered_axl_structure.L, calculated_shifted_structure.L
        )
        assert np.allclose(
            expected_centered_axl_structure.X, calculated_shifted_structure.X
        )

import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.no_op_excisor import (
    NoOpExcision, NoOpExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_number_of_lattice_parameters


class TestBaseEnvironmentExcision:
    @pytest.fixture(params=[1, 3, 5, 12])
    def topk_value(self, request):
        return request.param

    @pytest.fixture(params=[0.1, 0.5, 0.9])
    def uncertainty_threshold(self, request):
        return request.param

    @pytest.fixture
    def excisor_topk_arguments(self, topk_value):
        exc_arg = NoOpExcisionArguments(
            algorithm="test_abstract_method",
            uncertainty_threshold=None,
            excise_top_k_environment=topk_value,
        )
        return exc_arg

    @pytest.fixture
    def excisor_threshold_arguments(self, uncertainty_threshold):
        exc_arg = NoOpExcisionArguments(
            algorithm="test_abstract_method",
            uncertainty_threshold=uncertainty_threshold,
            excise_top_k_environment=None,
        )
        return exc_arg

    @pytest.fixture
    def number_of_atoms(self):
        return 20

    @pytest.fixture
    def uncertainty_per_atom(self, number_of_atoms):
        return np.random.random(number_of_atoms)

    def test_uncertainty_threshold_atom_selection(
        self, uncertainty_threshold, excisor_threshold_arguments, uncertainty_per_atom
    ):
        uncertainties, atom_idx = [], []
        for idx, u in enumerate(uncertainty_per_atom):
            if u > uncertainty_threshold:
                atom_idx.append(idx)
                uncertainties.append(u)
        # sort on uncertainty (highest to lowest)
        sorted_pairs = sorted(
            zip(uncertainties, atom_idx), key=lambda item: item[0], reverse=True
        )
        expected_atom_idx = [item[1] for item in sorted_pairs]

        base_excisor = NoOpExcision(excisor_threshold_arguments)
        calculated_atom_idx = base_excisor.select_central_atoms(uncertainty_per_atom)

        assert np.array_equal(calculated_atom_idx, np.array(expected_atom_idx))

    def test_uncertainty_topk_atom_selection(
        self, topk_value, excisor_topk_arguments, uncertainty_per_atom, number_of_atoms
    ):
        uncertainties = uncertainty_per_atom.tolist()
        atom_idx = range(number_of_atoms)

        # sort on uncertainty (highest to lowest)
        _, expected_atom_idx = zip(*sorted(zip(uncertainties, atom_idx), reverse=True))
        expected_atom_idx = expected_atom_idx[:topk_value]

        base_excisor = NoOpExcision(excisor_topk_arguments)
        calculated_atom_idx = base_excisor.select_central_atoms(uncertainty_per_atom)

        assert np.array_equal(calculated_atom_idx, np.array(expected_atom_idx))

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

    def test_center_structure(
        self, number_of_atoms, uncentered_axl_structure, excisor_threshold_arguments
    ):
        base_excisor = NoOpExcision(excisor_threshold_arguments)
        for central_atom_idx in range(number_of_atoms):
            expected_shifted_coordinates = self.calculated_centered_coordinates(
                central_atom_idx, uncentered_axl_structure.X
            )
            calculated_shifted_structure = base_excisor.center_structure(
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

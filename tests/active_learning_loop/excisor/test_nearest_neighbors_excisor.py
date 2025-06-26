import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.nearest_neighbors_excisor import (
    NearestNeighborsExcision, NearestNeighborsExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from tests.active_learning_loop.excisor.base_test_excision import \
    BaseTestExcision


class TestNearestNeighborsExcision(BaseTestExcision):
    @pytest.fixture(params=[1, 5, 10])
    def number_of_neighbors(self, request):
        return request.param

    @pytest.fixture
    def excisor(self, number_of_neighbors):
        parameters = NearestNeighborsExcisionArguments(number_of_neighbors=number_of_neighbors)
        return NearestNeighborsExcision(parameters)

    @pytest.fixture
    def expected_excised_environment(self,
                                     structure_axl,
                                     number_of_neighbors,
                                     central_atom_idx,
                                     atom_cartesian_positions,
                                     basis_vectors,
                                     ):
        central_atom_position = atom_cartesian_positions[central_atom_idx, :]
        box_dimensions = np.diag(basis_vectors)[np.newaxis, :]
        atoms_relative_coordinates_in_environment = []
        atoms_species_in_environment = []
        distances_squared = []
        for idx, atom_position in enumerate(atom_cartesian_positions):
            separation_between_atoms = atom_position - central_atom_position
            separation_squared_between_atoms = np.minimum(
                (separation_between_atoms**2),
                (separation_between_atoms - box_dimensions) ** 2,
            )
            separation_squared_between_atoms = np.minimum(
                separation_squared_between_atoms,
                (separation_between_atoms + box_dimensions) ** 2,
            )
            distances_squared.append(separation_squared_between_atoms.sum(axis=-1))
        # find the N smallest values
        closest_atoms = sorted(
            range(len(distances_squared)), key=lambda i: distances_squared[i]
        )
        closest_atoms = closest_atoms[: number_of_neighbors + 1]
        for idx in closest_atoms:
            atoms_relative_coordinates_in_environment.append(structure_axl.X[idx, :])
            atoms_species_in_environment.append(structure_axl.A[idx])
        expected_environment = AXL(
            A=np.stack(atoms_species_in_environment),
            X=np.stack(atoms_relative_coordinates_in_environment),
            L=structure_axl.L,
        )
        return expected_environment

    @pytest.fixture
    def expected_excised_atom_index(self):
        return 0

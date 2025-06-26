import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.spherical_excisor import (
    SphericalExcision, SphericalExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from tests.active_learning_loop.excisor.base_test_excision import \
    BaseTestExcision


class TestSphericalExcision(BaseTestExcision):
    @pytest.fixture(params=[1.2, 2.3])
    def radial_cutoff(self, request):
        return request.param

    @pytest.fixture
    def excisor(self, radial_cutoff):
        parameters = SphericalExcisionArguments(radial_cutoff=radial_cutoff)
        return SphericalExcision(parameters)

    @pytest.fixture
    def expected_excised_atom_index(self):
        # TODO
        return 0

    @pytest.fixture
    def expected_excised_environment(self, structure_axl, radial_cutoff, central_atom_idx,
                                     atom_cartesian_positions, basis_vectors):
        central_atom_position = atom_cartesian_positions[central_atom_idx, :]
        box_dimensions = np.diag(basis_vectors)[np.newaxis, :]
        atoms_positions_in_environment = []
        atoms_species_in_environment = []
        all_distances_squared = []
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
            distance_squared = separation_squared_between_atoms.sum(axis=-1)
            if distance_squared < radial_cutoff**2:
                all_distances_squared.append(distance_squared.item())
                atoms_positions_in_environment.append(structure_axl.X[idx])
                atoms_species_in_environment.append(structure_axl.A[idx])
        sorted_idx = np.argsort(all_distances_squared).tolist()
        expected_environment = AXL(
            A=np.stack(np.array(atoms_species_in_environment)[sorted_idx]),
            X=np.stack(np.array(atoms_positions_in_environment)[sorted_idx]),
            L=structure_axl.L,
        )
        return expected_environment

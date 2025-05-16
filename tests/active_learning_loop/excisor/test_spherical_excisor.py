import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.spherical_excisor import (
    SphericalExcision, SphericalExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


class TestSphericalExcision:
    @pytest.fixture(params=[1.0, 2.0])
    def radial_cutoff(self, request):
        return request.param

    @pytest.fixture(params=[1, 2, 3])
    def topk(self, request):
        return request.param

    @pytest.fixture
    def excisor_parameters(self, radial_cutoff, topk):
        exc_params = SphericalExcisionArguments(
            excise_top_k_environment=topk, radial_cutoff=radial_cutoff
        )
        return exc_params

    @pytest.fixture
    def spherical_excisor(self, excisor_parameters):
        return SphericalExcision(excisor_parameters)

    @pytest.fixture
    def number_of_atoms(self):
        return 48

    @pytest.fixture(params=[1, 2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture
    def basis_vectors(self, spatial_dimension):
        box_size = np.random.random((spatial_dimension,)) * 4
        return np.diag(box_size)

    @pytest.fixture
    def lattice_parameters(self, spatial_dimension, basis_vectors):
        lp = np.concatenate(
            (
                np.diag(basis_vectors),
                np.zeros(int(spatial_dimension * (spatial_dimension - 1) / 2)),
            )
        )
        return lp

    @pytest.fixture
    def atom_relative_positions(self, number_of_atoms, spatial_dimension):
        return np.random.random((number_of_atoms, spatial_dimension))

    @pytest.fixture
    def atom_cartesian_positions(self, atom_relative_positions, basis_vectors):
        return np.matmul(atom_relative_positions, basis_vectors)

    @pytest.fixture
    def atom_species(self, number_of_atoms):
        return np.arange(number_of_atoms)

    @pytest.fixture
    def structure_axl(self, atom_species, atom_relative_positions, lattice_parameters):
        struct_axl = AXL(
            A=atom_species, X=atom_relative_positions, L=lattice_parameters
        )
        return struct_axl

    @pytest.fixture
    def atom_uncertainty(self, number_of_atoms):
        return np.random.random((number_of_atoms,))

    @pytest.mark.parametrize("central_atom_idx", [1, 2, 3])
    def test_excise_one_environment(
        self,
        spherical_excisor,
        structure_axl,
        radial_cutoff,
        central_atom_idx,
        atom_cartesian_positions,
        basis_vectors,
    ):
        central_atom_position = atom_cartesian_positions[central_atom_idx, :]
        box_dimensions = np.diag(basis_vectors)[np.newaxis, :]
        atoms_positions_in_environment = []
        atoms_species_in_environment = []
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
                atoms_positions_in_environment.append(structure_axl.X[idx])
                atoms_species_in_environment.append(structure_axl.A[idx])
        expected_environment = AXL(
            A=np.stack(atoms_species_in_environment),
            X=np.stack(atoms_positions_in_environment),
            L=structure_axl.L,
        )

        calculated_environment = spherical_excisor._excise_one_environment(
            structure_axl, central_atom_idx
        )

        assert np.array_equal(expected_environment.A, calculated_environment.A)
        assert np.array_equal(expected_environment.X, calculated_environment.X)

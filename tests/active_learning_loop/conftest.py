import numpy as np
import pymatgen
import pytest
from pymatgen.core import Lattice, Structure

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@pytest.fixture()
def list_element_symbols():
    return ['Ca', 'Si', 'H', 'C']


@pytest.fixture()
def list_elements(list_element_symbols):
    return [pymatgen.core.Element(symbol) for symbol in list_element_symbols]


@pytest.fixture()
def structure(list_element_symbols):
    number_of_atoms = 32
    lattice = Lattice(matrix=np.diag([10, 12, 14]))
    relative_coordinates = np.random.rand(number_of_atoms, 3)

    species = np.random.choice(list_element_symbols, size=number_of_atoms, replace=True)

    structure = Structure(lattice=lattice,
                          species=species,
                          coords=relative_coordinates,
                          coords_are_cartesian=False)
    return structure


@pytest.fixture()
def expected_sorted_list_element_symbols():
    return ['H', 'C', 'Si', 'Ca']


class BaseTestAxlStructure:

    @pytest.fixture
    def number_of_atoms(self):
        return 48

    @pytest.fixture
    def element_list(self):
        return ['H', 'C', 'Si', 'Ca']

    @pytest.fixture
    def num_atom_types(self, element_list):
        return len(element_list)

    @pytest.fixture(params=[1, 2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture
    def basis_vectors(self, spatial_dimension):
        box_size = 22 + np.random.random((spatial_dimension,))
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
    def atom_relative_coordinates(self, number_of_atoms, spatial_dimension):
        return np.random.random((number_of_atoms, spatial_dimension))

    @pytest.fixture
    def atom_cartesian_positions(self, atom_relative_coordinates, basis_vectors):
        return np.matmul(atom_relative_coordinates, basis_vectors)

    @pytest.fixture
    def atom_species(self, num_atom_types, number_of_atoms):
        return np.random.randint(low=0, high=num_atom_types, size=(number_of_atoms,))

    @pytest.fixture
    def structure_axl(self, atom_species, atom_relative_coordinates, lattice_parameters):
        struct_axl = AXL(
            A=atom_species, X=atom_relative_coordinates, L=lattice_parameters
        )
        return struct_axl

    @pytest.fixture(params=range(48))
    def central_atom_idx(self, request):
        return request.param

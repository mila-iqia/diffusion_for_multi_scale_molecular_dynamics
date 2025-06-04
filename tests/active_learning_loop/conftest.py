import numpy as np
import pymatgen
import pytest
from pymatgen.core import Lattice, Structure


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

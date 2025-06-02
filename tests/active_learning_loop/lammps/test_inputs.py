import numpy as np
import pymatgen
import pytest
from pymatgen.core import Lattice, Structure

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.inputs import (
    generate_named_elements_blocks, sort_elements_by_atomic_mass)


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


@pytest.fixture()
def expected_group_block(expected_sorted_list_element_symbols):
    number_of_elements = len(expected_sorted_list_element_symbols)
    lines = ""
    for group_id in np.arange(1, number_of_elements + 1):
        symbol = expected_sorted_list_element_symbols[group_id - 1]
        lines += f"\ngroup {symbol} type {group_id}"

    return lines


@pytest.fixture()
def expected_mass_block(expected_sorted_list_element_symbols):
    number_of_elements = len(expected_sorted_list_element_symbols)
    lines = ""
    for group_id in np.arange(1, number_of_elements + 1):
        symbol = expected_sorted_list_element_symbols[group_id - 1]
        mass = pymatgen.core.Element(symbol).atomic_mass.real
        lines += f"\nmass {group_id} {mass}"
    return lines


@pytest.fixture()
def expected_elements_string(expected_sorted_list_element_symbols):
    return " ".join(expected_sorted_list_element_symbols)


def test_sort_elements_by_atomic_mass(list_elements, expected_sorted_list_element_symbols):
    computed_sorted_list_elements = sort_elements_by_atomic_mass(list_elements)
    computed_sorted_list_element_symbols = [element.symbol for element in computed_sorted_list_elements]
    assert computed_sorted_list_element_symbols == expected_sorted_list_element_symbols


def test_generate_named_elements_blocks(structure, expected_group_block, expected_mass_block, expected_elements_string):
    group_block, mass_block, elements_string = generate_named_elements_blocks(structure)
    assert group_block == expected_group_block
    assert mass_block == expected_mass_block
    assert elements_string == expected_elements_string

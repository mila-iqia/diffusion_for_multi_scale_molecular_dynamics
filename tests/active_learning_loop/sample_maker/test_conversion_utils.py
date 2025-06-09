import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.conversion_utils import (
    convert_axl_to_structure, convert_structure_to_axl)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_unit_cell_to_lattice_parameters


@pytest.fixture()
def number_of_atoms():
    return 32


@pytest.fixture()
def unique_symbols():
    return ['Ca', 'Si', 'H', 'C']


@pytest.fixture()
def box_dimensions():
    return 10. + 10. * np.random.rand(3)


@pytest.fixture()
def species(unique_symbols, number_of_atoms):
    return np.random.choice(unique_symbols, number_of_atoms, replace=True)


@pytest.fixture()
def relative_coordinates(number_of_atoms):
    return np.random.rand(number_of_atoms, 3)


@pytest.fixture()
def structure(box_dimensions, species, relative_coordinates):

    lattice = Lattice(matrix=np.diag(box_dimensions))
    structure = Structure(lattice=lattice,
                          species=species,
                          coords=relative_coordinates,
                          coords_are_cartesian=False)
    return structure


@pytest.fixture()
def axl_structure(box_dimensions, species, relative_coordinates):

    axl_structure = AXL(A=species,
                        X=relative_coordinates,
                        L=map_unit_cell_to_lattice_parameters(unit_cell=np.diag(box_dimensions), engine="numpy"))

    return axl_structure


def test_convert_axl_to_structure(axl_structure, structure):
    computed_structure = convert_axl_to_structure(axl_structure)
    assert computed_structure == structure


def test_convert_structure_to_axl(axl_structure, structure):
    computed_axl_structure = convert_structure_to_axl(structure)

    np.testing.assert_allclose(computed_axl_structure.L, axl_structure.L)
    np.testing.assert_allclose(computed_axl_structure.X, axl_structure.X)
    assert np.all(computed_axl_structure.A == axl_structure.A)

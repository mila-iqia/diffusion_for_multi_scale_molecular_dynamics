import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.structure_converter import \
    StructureConverter
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_unit_cell_to_lattice_parameters


class TestStructureConverter:
    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        np.random.seed(34534)

    @pytest.fixture()
    def number_of_atoms(self):
        return 32

    @pytest.fixture()
    def unique_symbols(self):
        return ['Ca', 'Si', 'H', 'C']

    @pytest.fixture()
    def box_dimensions(self):
        return 10. + 10. * np.random.rand(3)

    @pytest.fixture()
    def species(self, unique_symbols, number_of_atoms):
        return np.random.choice(unique_symbols, number_of_atoms, replace=True)

    @pytest.fixture()
    def element_ids(self, unique_symbols, species):
        element_types = ElementTypes(unique_symbols)
        return np.array([element_types.get_element_id(s) for s in species])

    @pytest.fixture()
    def relative_coordinates(self, number_of_atoms):
        return np.random.rand(number_of_atoms, 3)

    @pytest.fixture()
    def structure(self, box_dimensions, species, relative_coordinates):

        lattice = Lattice(matrix=np.diag(box_dimensions))
        structure = Structure(lattice=lattice,
                              species=species,
                              coords=relative_coordinates,
                              coords_are_cartesian=False)
        return structure

    @pytest.fixture()
    def axl_structure(self, box_dimensions, element_ids, relative_coordinates):

        axl_structure = AXL(A=element_ids,
                            X=relative_coordinates,
                            L=map_unit_cell_to_lattice_parameters(unit_cell=np.diag(box_dimensions), engine="numpy"))

        return axl_structure

    @pytest.fixture()
    def structure_converter(self, unique_symbols):
        return StructureConverter(list_of_element_symbols=unique_symbols)

    def test_convert_axl_to_structure(self, structure_converter, axl_structure, structure):
        computed_structure = structure_converter.convert_axl_to_structure(axl_structure)
        assert computed_structure == structure

    def test_convert_structure_to_axl(self, structure_converter, axl_structure, structure):
        computed_axl_structure = structure_converter.convert_structure_to_axl(structure)

        np.testing.assert_allclose(computed_axl_structure.L, axl_structure.L)
        np.testing.assert_allclose(computed_axl_structure.X, axl_structure.X)
        np.testing.assert_allclose(computed_axl_structure.A, axl_structure.A)

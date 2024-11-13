import einops
import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_calculator import (
    LammpsCalculator, LammpsOracleParameters)


@pytest.mark.not_on_github
class TestLammpsCalculator:

    @pytest.fixture(scope="class", autouse=True)
    def set_seed(self):
        """Set the random seed."""
        np.random.seed(2311331423)

    @pytest.fixture()
    def spatial_dimension(self):
        return 3

    @pytest.fixture(params=[8, 12, 16])
    def num_atoms(self, request):
        return request.param

    @pytest.fixture()
    def acell(self):
        return 5.5

    @pytest.fixture()
    def box(self, spatial_dimension, acell):
        return np.diag(spatial_dimension * [acell])

    @pytest.fixture()
    def cartesian_positions(self, num_atoms, spatial_dimension, box):
        x = np.random.rand(num_atoms, spatial_dimension)
        return einops.einsum(box, x, "d1 d2, natoms d2 -> natoms d1")

    @pytest.fixture(params=[1, 2])
    def number_of_unique_elements(self, request):
        return request.param

    @pytest.fixture()
    def unique_elements(self, number_of_unique_elements):
        if number_of_unique_elements == 1:
            return ['Si']
        elif number_of_unique_elements == 2:
            return ['Si', 'Ge']

    @pytest.fixture()
    def lammps_oracle_parameters(self, number_of_unique_elements):
        if number_of_unique_elements == 1:
            return LammpsOracleParameters(sw_coeff_filename='Si.sw')
        elif number_of_unique_elements == 2:
            return LammpsOracleParameters(sw_coeff_filename='SiGe.sw')

    @pytest.fixture()
    def element_types(self, unique_elements):
        return ElementTypes(unique_elements)

    @pytest.fixture()
    def atom_types(self, element_types, num_atoms):
        return np.random.choice(element_types.element_ids, num_atoms, replace=True)

    @pytest.fixture()
    def calculator(self, element_types, lammps_oracle_parameters, tmp_path):
        calculator = LammpsCalculator(lammps_oracle_parameters=lammps_oracle_parameters,
                                      element_types=element_types,
                                      tmp_work_dir=tmp_path)
        return calculator

    def test_calculator(self, calculator, element_types, cartesian_positions, box, atom_types, tmp_path):

        energy, forces = calculator.compute_energy_and_forces(cartesian_positions, box, atom_types)

        np.testing.assert_allclose(cartesian_positions, forces[['x', 'y', 'z']].values, rtol=1e-5)

        expected_atoms = [element_types.get_element(id) for id in atom_types]
        computed_atoms = forces['element'].to_list()
        assert expected_atoms == computed_atoms

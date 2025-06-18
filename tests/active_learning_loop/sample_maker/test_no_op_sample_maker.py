import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.no_op_excisor import (
    NoOpExcision, NoOpExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_noop_sample_maker import (  # noqa
    ExciseAndNoOpSampleMaker, ExciseAndNoOpSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.no_op_sample_maker import (
    NoOpSampleMaker, NoOpSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_number_of_lattice_parameters


class TestNoOpSampleMaker:

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        np.random.seed(34523423)

    @pytest.fixture(params=[1, 2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture
    def number_of_atoms(self):
        return 32

    @pytest.fixture
    def element_list(self):
        return ["fire", "earth", "air", "water"]

    @pytest.fixture
    def sample_maker_arguments(self, spatial_dimension, element_list):
        return NoOpSampleMakerArguments(element_list=element_list)

    @pytest.fixture(params=['uncertainty_threshold', 'excise_top_k_environment'])
    def excision_strategy(self, request):
        return request.param

    @pytest.fixture()
    def uncertainty_threshold(self) -> float:
        return 0.001

    @pytest.fixture()
    def excise_top_k_environment(self):
        return 4

    @pytest.fixture
    def uncertainty_per_atom(self, number_of_atoms, uncertainty_threshold):
        return 2 * np.random.rand(number_of_atoms) * uncertainty_threshold

    @pytest.fixture
    def excision_parameter_dictionary(self, excision_strategy, uncertainty_threshold, excise_top_k_environment):
        match excision_strategy:
            case "uncertainty_threshold":
                return dict(uncertainty_threshold=uncertainty_threshold)
            case "excise_top_k_environment":
                return dict(excise_top_k_environment=excise_top_k_environment)
            case _:
                raise NotImplementedError("Unknown excision strategy.")

    @pytest.fixture()
    def excisor(self, excision_parameter_dictionary):
        return NoOpExcision(NoOpExcisionArguments(**excision_parameter_dictionary))

    @pytest.fixture
    def noop_sample_maker(self, sample_maker_arguments, excisor):
        return NoOpSampleMaker(sample_maker_arguments, excisor)

    @pytest.fixture
    def axl_structure(self, number_of_atoms, element_list, spatial_dimension):
        atom_types = np.random.randint(0, len(element_list), number_of_atoms)
        coordinates = np.random.rand(number_of_atoms, spatial_dimension)
        num_lattice_parameters = get_number_of_lattice_parameters(spatial_dimension)
        lattice_params = np.concatenate(
            [
                np.random.rand(spatial_dimension),
                np.zeros((num_lattice_parameters - spatial_dimension,)),
            ]
        )
        return AXL(A=atom_types, X=coordinates, L=lattice_params)

    @pytest.fixture
    def expected_active_indices(self, axl_structure, uncertainty_per_atom, excision_parameter_dictionary):
        if 'uncertainty_threshold' in excision_parameter_dictionary:
            threshold = excision_parameter_dictionary['uncertainty_threshold']
            expected_active_indices = np.where(uncertainty_per_atom > threshold)[0]
        elif 'excise_top_k_environment' in excision_parameter_dictionary:
            top_k = excision_parameter_dictionary['excise_top_k_environment']
            expected_active_indices = np.argsort(uncertainty_per_atom)[-top_k:]
        else:
            raise NotImplementedError("Unknown excision strategy.")

        return expected_active_indices

    def test_make_samples(
        self,
        noop_sample_maker,
        axl_structure,
        uncertainty_per_atom,
        expected_active_indices
    ):
        list_samples, list_active_indices, extra_info = (
            noop_sample_maker.make_samples(axl_structure, uncertainty_per_atom))

        assert len(list_samples) == 1
        assert len(list_active_indices) == 1
        assert len(extra_info) == 1

        sample_axl_structure = list_samples[0]
        active_indices = list_active_indices[0]

        np.testing.assert_allclose(sample_axl_structure.A, axl_structure.A)
        np.testing.assert_allclose(sample_axl_structure.X, axl_structure.X)
        np.testing.assert_allclose(sample_axl_structure.L, axl_structure.L)

        assert set(active_indices) == set(expected_active_indices)

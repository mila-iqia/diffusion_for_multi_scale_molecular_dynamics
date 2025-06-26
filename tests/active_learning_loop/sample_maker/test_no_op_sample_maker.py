import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_noop_sample_maker import (  # noqa
    ExciseAndNoOpSampleMaker, ExciseAndNoOpSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.no_op_sample_maker import (
    NoOpSampleMaker, NoOpSampleMakerArguments)
from tests.active_learning_loop.sample_maker.base_test_sample_maker import \
    BaseTestSampleMaker


class TestNoOpSampleMaker(BaseTestSampleMaker):

    @pytest.fixture()
    def sample_maker(self, atom_selector, element_list):
        return NoOpSampleMaker(sample_maker_arguments=NoOpSampleMakerArguments(element_list=element_list),
                               atom_selector=atom_selector)

    @pytest.fixture
    def list_expected_active_environment_indices(self, atom_selector, uncertainty_per_atom):
        return [atom_selector.select_central_atoms(uncertainty_per_atom)]

    @pytest.fixture
    def list_expected_sample_structures(self, structure_axl):
        return [structure_axl]

    def test_make_samples(
        self,
        sample_maker,
        structure_axl,
        uncertainty_per_atom,
        list_expected_sample_structures,
        list_expected_active_environment_indices,
    ):
        list_sample_structures, list_active_environment_indices, _ = (
            sample_maker.make_samples(structure_axl, uncertainty_per_atom))

        assert len(list_sample_structures) == len(list_expected_sample_structures)
        assert len(list_expected_active_environment_indices) == len(list_active_environment_indices)

        for sample_structure, expected_sample_structure in zip(list_sample_structures, list_expected_sample_structures):
            np.testing.assert_allclose(sample_structure.A, expected_sample_structure.A)
            np.testing.assert_allclose(sample_structure.X, expected_sample_structure.X)
            np.testing.assert_allclose(sample_structure.L, expected_sample_structure.L)

        for indices, expected_indices in zip(list_active_environment_indices, list_expected_active_environment_indices):
            assert set(indices) == set(expected_indices)

import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import (
    NoOpEnvironmentExcision, NoOpEnvironmentExcisionArguments)


class TestBaseEnvironmentExcision:
    @pytest.fixture(params=[1, 3, 5, 12])
    def topk_value(self, request):
        return request.param

    @pytest.fixture(params=[0.1, 0.5, 0.9])
    def uncertainty_threshold(self, request):
        return request.param

    @pytest.fixture
    def excisor_topk_arguments(self, topk_value):
        exc_arg = NoOpEnvironmentExcisionArguments(
            algorithm="test_abstract_method",
            uncertainty_threshold=None,
            excise_top_k_environment=topk_value,
        )
        return exc_arg

    @pytest.fixture
    def excisor_threshold_arguments(self, uncertainty_threshold):
        exc_arg = NoOpEnvironmentExcisionArguments(
            algorithm="test_abstract_method",
            uncertainty_threshold=uncertainty_threshold,
            excise_top_k_environment=None,
        )
        return exc_arg

    @pytest.fixture
    def number_of_atoms(self):
        return 20

    @pytest.fixture
    def uncertainty_per_atom(self, number_of_atoms):
        return np.random.random(number_of_atoms)

    def test_uncertainty_threshold_atom_selection(
        self, uncertainty_threshold, excisor_threshold_arguments, uncertainty_per_atom
    ):
        uncertainties, atom_idx = [], []
        for idx, u in enumerate(uncertainty_per_atom):
            if u > uncertainty_threshold:
                atom_idx.append(idx)
                uncertainties.append(u)
        # sort on uncertainty (highest to lowest)
        sorted_pairs = sorted(zip(uncertainties, atom_idx), key=lambda item: item[0], reverse=True)
        expected_atom_idx = [item[1] for item in sorted_pairs]

        base_excisor = NoOpEnvironmentExcision(excisor_threshold_arguments)
        calculated_atom_idx = base_excisor.select_central_atoms(uncertainty_per_atom)

        assert np.array_equal(calculated_atom_idx, np.array(expected_atom_idx))

    def test_uncertainty_topk_atom_selection(
        self, topk_value, excisor_topk_arguments, uncertainty_per_atom, number_of_atoms
    ):
        uncertainties = uncertainty_per_atom.tolist()
        atom_idx = range(number_of_atoms)

        # sort on uncertainty (highest to lowest)
        _, expected_atom_idx = zip(*sorted(zip(uncertainties, atom_idx), reverse=True))
        expected_atom_idx = expected_atom_idx[:topk_value]

        base_excisor = NoOpEnvironmentExcision(excisor_topk_arguments)
        calculated_atom_idx = base_excisor.select_central_atoms(uncertainty_per_atom)

        assert np.array_equal(calculated_atom_idx, np.array(expected_atom_idx))

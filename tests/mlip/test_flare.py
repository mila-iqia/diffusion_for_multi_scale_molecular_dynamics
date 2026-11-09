import itertools

import numpy as np
import pytest


def get_all_flag_combinations():
    """All combinations of the four optimize-flags, excluding the all-false case."""
    return [list(flags) for flags in itertools.product([True, False], repeat=4) if any(flags)]


@pytest.mark.requires_flare
class TestFlareHyperparameterOptimizer:

    @pytest.fixture(params=get_all_flag_combinations())
    def training_flags(self, request):
        return request.param

    @pytest.fixture
    def translator(self, training_flags):
        from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_hyperparameter_optimizer import \
            HyperparameterTranslator
        return HyperparameterTranslator(*training_flags)

    @pytest.fixture
    def starting_hyperparameters(self):
        return np.random.rand(4)

    @pytest.fixture
    def minimization_input(self, training_flags):
        return np.random.rand(int(np.sum(training_flags)))

    @pytest.fixture
    def expected_hyperparameters(self, starting_hyperparameters, minimization_input, training_flags):
        expected_hyperparameters = 1.0 * starting_hyperparameters
        position = 0
        for index, flag in enumerate(training_flags):
            if flag:
                expected_hyperparameters[index] = minimization_input[position]
                position += 1
        return expected_hyperparameters

    def test_generate_sgp_hyperparameters_from_minimization_inputs(
        self, translator, starting_hyperparameters, minimization_input, expected_hyperparameters
    ):
        """Only the flagged hyperparameters are overwritten by the minimizer's inputs."""
        computed_hyperparameters = translator.generate_sgp_hyperparameters_from_minimization_inputs(
            starting_hyperparameters, minimization_input
        )
        np.testing.assert_allclose(computed_hyperparameters, expected_hyperparameters)

import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_hyperparameter_optimizer import (
    FlareHyperparametersOptimizer, FlareOptimizerConfiguration,
    HyperparameterTranslator)
from tests.active_learning_loop.trainer.base_test_flare import BaseTestFlare


class TestFlareHyperparameterOptimizer(BaseTestFlare):

    @pytest.fixture(params=[True])
    def optimize_sigma(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def optimize_sigma_e(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def optimize_sigma_f(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def optimize_sigma_s(self, request):
        return request.param

    @pytest.fixture()
    def number_of_parameters_to_optimize(self, optimize_sigma, optimize_sigma_e, optimize_sigma_f, optimize_sigma_s):
        return optimize_sigma + optimize_sigma_e + optimize_sigma_f + optimize_sigma_s

    @pytest.fixture()
    def starting_hyperparameters(self):
        return np.random.rand(4)

    @pytest.fixture()
    def minimization_input(self, number_of_parameters_to_optimize):
        return np.random.rand(number_of_parameters_to_optimize)

    @pytest.fixture()
    def expected_hyperparameters(self, starting_hyperparameters, minimization_input,
                                 optimize_sigma, optimize_sigma_e, optimize_sigma_f, optimize_sigma_s):

        new_parameters = 1.0 * starting_hyperparameters
        position = 0
        for idx, flag in enumerate([optimize_sigma, optimize_sigma_e, optimize_sigma_f, optimize_sigma_s]):
            if flag:
                new_parameters[idx] = minimization_input[position]
                position += 1

        return new_parameters

    @pytest.fixture()
    def translator(self, optimize_sigma, optimize_sigma_e, optimize_sigma_f, optimize_sigma_s):
        return HyperparameterTranslator(optimize_sigma, optimize_sigma_e, optimize_sigma_f, optimize_sigma_s)

    def test_generate_sgp_hyperparameters_from_minimization_inputs(self, translator, starting_hyperparameters,
                                                                   minimization_input, expected_hyperparameters):
        computed_hyperparameters = (
            translator.generate_sgp_hyperparameters_from_minimization_inputs(starting_hyperparameters,
                                                                             minimization_input))
        np.testing.assert_allclose(expected_hyperparameters, computed_hyperparameters)

    @pytest.fixture(params=["BFGS", "L-BFGS-B", "nelder-mead"])
    def optimization_method(self, request):
        return request.param

    @pytest.fixture
    def number_of_iterations(self):
        return 20

    @pytest.fixture
    def configuration(self, optimization_method, number_of_iterations,
                      optimize_sigma, optimize_sigma_e, optimize_sigma_f, optimize_sigma_s):
        return FlareOptimizerConfiguration(optimization_method=optimization_method,
                                           max_optimization_iterations=number_of_iterations,
                                           optimize_sigma=optimize_sigma,
                                           optimize_sigma_e=optimize_sigma_e,
                                           optimize_sigma_f=optimize_sigma_f,
                                           optimize_sigma_s=optimize_sigma_s)

    @pytest.fixture()
    def optimizer(self, configuration):
        return FlareHyperparametersOptimizer(flare_optimizer_configuration=configuration)

    def test_smoke_flare_hyperparameter_optimizer(self, optimizer, flare_configuration,
                                                  number_of_parameters_to_optimize, labelled_structure,
                                                  active_environment_indices, number_of_iterations):

        flare_trainer = self.instantiate_flare_trainer(flare_configuration,
                                                       labelled_structure,
                                                       active_environment_indices)

        results, history_df = optimizer.train(flare_trainer.sgp_model)

import importlib.util

import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_hyperparameter_optimizer import \
    FlareHyperparametersOptimizer
from tests.active_learning_loop.trainer.base_test_flare import BaseTestFlare


@pytest.mark.skipif(importlib.util.find_spec("flare_pp") is None, reason="FLARE is not installed")
class TestFlareHyperparameterOptimizer(BaseTestFlare):

    @pytest.fixture(params=["BFGS", "L-BFGS-B", "nelder-mead"])
    def method(self, request):
        return request.param

    @pytest.fixture
    def number_of_iterations(self):
        return 20

    @pytest.fixture()
    def optimizer(self, method, number_of_iterations):
        minimize_options = {"disp": False, "ftol": 1e-8, "gtol": 1e-8, "maxiter": number_of_iterations}
        optimizer = FlareHyperparametersOptimizer(method, minimize_options)
        return optimizer

    def test_smoke_flare_hyperparameter_optimizer(self, optimizer, flare_configuration,
                                                  labelled_structure, active_environment_indices, number_of_iterations):

        flare_trainer = self.instantiate_flare_trainer(flare_configuration,
                                                       labelled_structure,
                                                       active_environment_indices)

        results, history_df = optimizer.train(flare_trainer.sgp_model)

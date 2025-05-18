from typing import Dict

from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp.sparse_gp import compute_negative_likelihood_grad_stable, compute_negative_likelihood

import numpy as np

from scipy.optimize import minimize

class OptimizationTracker:
    def __init__(self):
        self.values = []

    def callback(self, intermediate_result):
        value = intermediate_result.fun
        self.values.append(value)


class HyperparametersTrainer:
    """Simple class to drive the training of a sparse GP."""

    def __init__(self, method: str, minimize_options: Dict):
        assert method in ["BFGS", "L-BFGS-B", "nelder-mead"], f"Unknown method {method}"

        self._minimize_options = minimize_options
        self._method = method

        self._requires_gradient = True
        if  self._method == "nelder-mead":
            self._requires_gradient = False

    def _create_function_to_minimize_no_gradient(self, sparse_gp):
        def function_to_minimize(hyperparameters):
            return compute_negative_likelihood(hyperparameters, sparse_gp, print_vals=True)
        return function_to_minimize

    def _create_function_to_minimize_with_gradient(self, sparse_gp):
        sparse_gp.precompute_KnK()
        def function_to_minimize(hyperparameters):
            return compute_negative_likelihood_grad_stable(hyperparameters, sparse_gp, precomputed=True)
        return function_to_minimize

    def _create_function_to_minimize(self, sparse_gp):
            if self._requires_gradient:
                return self._create_function_to_minimize_with_gradient(sparse_gp)
            else:
                return self._create_function_to_minimize_no_gradient(sparse_gp)

    def train(self, sgp_model: SGP_Wrapper):

        function_to_minimize = self._create_function_to_minimize(sgp_model.sparse_gp)

        optimization_tracker = OptimizationTracker()

        initial_guess = sgp_model.sparse_gp.hyperparameters

        optimization_result = minimize(function_to_minimize,
                                       initial_guess,
                                       method=self._method,
                                       jac=self._requires_gradient,
                                       callback=optimization_tracker.callback,
                                       options=self._minimize_options)

        return optimization_result, np.array(optimization_tracker.values)

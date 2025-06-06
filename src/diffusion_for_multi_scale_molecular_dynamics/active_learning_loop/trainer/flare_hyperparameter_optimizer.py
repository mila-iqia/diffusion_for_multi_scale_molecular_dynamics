from typing import Dict, Tuple

import pandas as pd
from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp.sparse_gp import (compute_negative_likelihood,
                                      compute_negative_likelihood_grad_stable)
from flare_pp import SparseGP
from scipy.optimize import OptimizeResult, minimize


class OptimizationTracker:
    """Optimization tracker.

    This class will keep track of intermediate values os a scipy optimization
    algorithm through a callback method.
    """
    def __init__(self):
        """Init method."""
        self.rows = []

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history."""
        df = pd.DataFrame(self.rows)
        df.index.name = "iteration step"
        return df

    def callback(self, intermediate_result: OptimizeResult):
        """Callback."""
        # By inspection of the FLARE code (see file flare/src/flare_pp/bffs/sparse_gp.cpp), we can deduce that
        # there are 4 hyperparameters and that they are
        #   - sigma : the prefactor for the dot product kernel)
        #   - sigma_energy : the energy noise term
        #   - sigma_force : the force noise term
        #   - sigma_stress : the stress noise term
        # where "noise" means that the GP predicts f, and the real lalbel is y ~ f + z * noise.
        assert len(intermediate_result.x) == 4, "unexpected number of hyperparmeters."
        row = dict(sigma=intermediate_result.x[0],
                   sigma_energy=intermediate_result.x[1],
                   sigma_forces=intermediate_result.x[2],
                   sigma_stress=intermediate_result.x[3],
                   negative_log_likelihood=intermediate_result.fun)
        self.rows.append(row)


class FlareHyperparametersOptimizer:
    """Class to drive the training of a FLARE sparse GP."""

    def __init__(self, method: str, minimize_options: Dict):
        """Init method."""
        assert method in ["BFGS", "L-BFGS-B", "nelder-mead"], f"Unknown method {method}"

        self._minimize_options = minimize_options
        self._method = method

        self._requires_gradient = True
        if self._method == "nelder-mead":
            self._requires_gradient = False

    def _create_function_to_minimize_no_gradient(self, sparse_gp: SparseGP):
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

    def train(self, sgp_model: SGP_Wrapper) -> Tuple[OptimizeResult, pd.DataFrame]:
        """Train.

        This method trains the spare GP. Note that it has the SIDE EFFECT of
        modifying the hyperparameters of the sparse GP in the sgp_model object.

        Args:
            sgp_model : a FLARE spare GP wrapper that contains a spare GP to be trained.

        Returns:
            optimization_results: the optimization result output of scipy.minimize.
            history_df: a dataframe with the minimizing iterations.
        """
        function_to_minimize = self._create_function_to_minimize(sgp_model.sparse_gp)

        optimization_tracker = OptimizationTracker()

        initial_guess = sgp_model.sparse_gp.hyperparameters

        optimization_result = minimize(function_to_minimize,
                                       initial_guess,
                                       method=self._method,
                                       jac=self._requires_gradient,
                                       callback=optimization_tracker.callback,
                                       options=self._minimize_options)

        history_df = optimization_tracker.get_optimization_history()

        return optimization_result, history_df

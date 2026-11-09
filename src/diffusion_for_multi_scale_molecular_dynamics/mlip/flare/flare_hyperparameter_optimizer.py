from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp.sparse_gp import compute_negative_likelihood_grad_stable
from scipy.optimize import OptimizeResult, minimize


@dataclass(kw_only=True)
class FlareOptimizerConfiguration:
    """Flare Optimizer Configuration.

    Various parameters defining how the hyperparameters of the FLARE sparce Gaussian process
    should be optimized.
    """
    optimization_method: str = "BFGS"
    max_optimization_iterations: int = 100

    optimize_sigma: bool = True
    optimize_sigma_e: bool = True
    optimize_sigma_f: bool = True
    optimize_sigma_s: bool = True

    print: bool = False  # Should the scipy algorithm print progress to screen?
    ftol: float = 1e-3
    gtol: float = 1e-3

    def __post_init__(self):
        """Post init."""
        assert self.optimization_method in ["BFGS", "L-BFGS-B", "nelder-mead"], \
            f"Unknown optimization method {self.optimization_method}. Review input."

        assert self.max_optimization_iterations >= 0, "The number of iterations should be non-negative."


class FlareHyperparametersOptimizer:
    """Class to drive the training of a FLARE sparse GP."""

    @property
    def is_inactive(self) -> bool:
        """Are all trainable flags set to false?"""
        return np.sum(self._ordered_training_flags) == 0

    def __init__(self, flare_optimizer_configuration: FlareOptimizerConfiguration):
        """Init method."""
        self.flare_optimizer_configuration = flare_optimizer_configuration

        self._ordered_training_flags = [self.flare_optimizer_configuration.optimize_sigma,
                                        self.flare_optimizer_configuration.optimize_sigma_e,
                                        self.flare_optimizer_configuration.optimize_sigma_f,
                                        self.flare_optimizer_configuration.optimize_sigma_s]

        self._translator = HyperparameterTranslator(*self._ordered_training_flags)
        self._optimization_method = self.flare_optimizer_configuration.optimization_method

        self._requires_gradient = True
        if self._optimization_method == "nelder-mead":
            self._requires_gradient = False

    def _create_function_to_minimize(self, sparse_gp):
        sparse_gp.precompute_KnK()

        def function_to_minimize(minimization_input: np.array):
            starting_hyperparameters = 1.0 * sparse_gp.hyperparameters
            hyperparameters = (
                self._translator.generate_sgp_hyperparameters_from_minimization_inputs(starting_hyperparameters,
                                                                                       minimization_input))
            nll, grads = compute_negative_likelihood_grad_stable(hyperparameters, sparse_gp, precomputed=True)

            if self._requires_gradient:
                return nll, grads[self._ordered_training_flags]
            else:
                return nll

        return function_to_minimize

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
        initial_hyperparameters = sgp_model.sparse_gp.hyperparameters

        optimization_tracker = OptimizationTracker(initial_hyperparameters=initial_hyperparameters,
                                                   translator=self._translator)

        initial_guess = initial_hyperparameters[self._ordered_training_flags]

        options_dict = dict(maxiter=self.flare_optimizer_configuration.max_optimization_iterations,
                            disp=self.flare_optimizer_configuration.print,
                            ftol=self.flare_optimizer_configuration.ftol,
                            gtol=self.flare_optimizer_configuration.gtol)

        optimization_result = minimize(function_to_minimize,
                                       initial_guess,
                                       method=self._optimization_method,
                                       jac=self._requires_gradient,
                                       callback=optimization_tracker.callback,
                                       options=options_dict)

        history_df = optimization_tracker.get_optimization_history()

        return optimization_result, history_df


class HyperparameterTranslator:
    """Hyperparameter translator.

    This class is responsible for mapping the sgp hyperparameters to the inputs of the function to
    be minimized to obtain the negative log likelihood.
    """
    def __init__(self, optimize_sigma: bool, optimize_sigma_e: bool, optimize_sigma_f: bool, optimize_sigma_s: bool):
        """Init method."""
        self._ordered_training_flags = [optimize_sigma, optimize_sigma_e, optimize_sigma_f, optimize_sigma_s]

    def generate_sgp_hyperparameters_from_minimization_inputs(self, starting_hyperparameters: np.array,
                                                              minimization_input: np.array) -> np.array:
        """Generate SGP hyperparameters from minimization inputs."""
        # The minimization input may have a different dimension as the hyperparameters.
        new_inputs = np.zeros(len(starting_hyperparameters))
        new_inputs[self._ordered_training_flags] = minimization_input
        hyperparameters = np.where(self._ordered_training_flags, new_inputs, starting_hyperparameters)
        return hyperparameters


class OptimizationTracker:
    """Optimization tracker.

    This class will keep track of intermediate values os a scipy optimization
    algorithm through a callback method.
    """
    def __init__(self, initial_hyperparameters: np.array, translator: HyperparameterTranslator):
        """Init method."""
        self._initial_hyperparameters = initial_hyperparameters
        self._translator = translator

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
        # where "noise" means that the GP predicts f, and the real label is y ~ f + z * noise.
        minimization_input = intermediate_result.x
        hyperparameters = (
            self._translator.generate_sgp_hyperparameters_from_minimization_inputs(self._initial_hyperparameters,
                                                                                   minimization_input))
        row = dict(sigma=hyperparameters[0],
                   sigma_energy=hyperparameters[1],
                   sigma_forces=hyperparameters[2],
                   sigma_stress=hyperparameters[3],
                   negative_log_likelihood=intermediate_result.fun)

        self.rows.append(row)

import logging
from dataclasses import dataclass
from typing import Callable

import einops
import torch
import torchode as to
from torchode import Solution

from crystal_diffusion.generators.position_generator import (
    PositionGenerator, SamplingParameters)
from crystal_diffusion.models.score_networks.score_network import ScoreNetwork
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from crystal_diffusion.utils.sample_trajectory import (NoOpODESampleTrajectory,
                                                       ODESampleTrajectory)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ODESamplingParameters(SamplingParameters):
    """Hyper-parameters for diffusion sampling with the ode algorithm."""
    algorithm: str = 'ode'
    absolute_solver_tolerance: float = 1.0e-3  # the absolute error tolerance passed to the ODE solver.
    relative_solver_tolerance: float = 1.0e-2  # the relative error tolerance passed to the ODE solver.


class ExplodingVarianceODEPositionGenerator(PositionGenerator):
    """Exploding Variance ODE Position Generator.

    This class generates position samples by solving an ordinary differential equation (ODE).
    It assumes that the diffusion noise is parameterized in the 'Exploding Variance' scheme.
    """

    def __init__(self,
                 noise_parameters: NoiseParameters,
                 sampling_parameters: ODESamplingParameters,
                 sigma_normalized_score_network: ScoreNetwork,
                 ):
        """Init method.

        Args:
            noise_parameters : the diffusion noise parameters.
            sampling_parameters: the parameters needed for sampling.
            sigma_normalized_score_network : the score network to use for drawing samples.
        """
        self.t0 = 0.0  # The "initial diffusion time", corresponding to the physical distribution.
        self.tf = 1.0  # The "final diffusion time", corresponding to the uniform distribution.

        self.noise_parameters = noise_parameters
        self.sigma_normalized_score_network = sigma_normalized_score_network

        assert self.noise_parameters.total_time_steps >= 2, \
            "There must at least be two time steps in the noise parameters to define the limits t0 and tf."
        self.number_of_atoms = sampling_parameters.number_of_atoms
        self.spatial_dimension = sampling_parameters.spatial_dimension
        self.absolute_solver_tolerance = sampling_parameters.absolute_solver_tolerance
        self.relative_solver_tolerance = sampling_parameters.relative_solver_tolerance
        self.record_samples = sampling_parameters.record_samples

        if self.record_samples:
            self.sample_trajectory_recorder = ODESampleTrajectory()
        else:
            self.sample_trajectory_recorder = NoOpODESampleTrajectory()

    def _get_exploding_variance_sigma(self, times):
        """Get Exploding Variance Sigma.

        In the 'exploding variance' scheme, the noise is defined by

            sigma(t) = sigma_min^{1- t} x sigma_max^{t}

        Args:
            times : diffusion time

        Returns:
            sigmas: value of the noise parameter.
        """
        sigmas = self.noise_parameters.sigma_min ** (1.0 - times) * self.noise_parameters.sigma_max ** times
        return sigmas

    def _get_ode_prefactor(self, sigmas):
        """Get ODE prefactor.

        The ODE is given by
            dx = [-1/2 g(t)^2 x Score] dt
        with
            g(t)^2 = d sigma(t)^2 / dt

        We can rearrange the ODE to:

            dx = -[1/2 g(t)^2 / sigma] x sigma Score
                  --------v-----------
                       Prefactor.

        The prefactor is then given by

            Prefactor = d sigma(t) / dt

        Args:
            sigmas : the values of the noise parameters.

        Returns:
            ode prefactor: the prefactor in the ODE.
        """
        log_ratio = torch.log(torch.tensor(self.noise_parameters.sigma_max / self.noise_parameters.sigma_min))
        ode_prefactor = log_ratio * sigmas
        return ode_prefactor

    def generate_ode_term(self, unit_cell: torch.Tensor) -> Callable:
        """Generate the ode_term needed to compute the ODE solution."""

        def ode_term(times: torch.Tensor, flat_relative_coordinates: torch.Tensor) -> torch.Tensor:
            """ODE term.

            This function is in the format required by the ODE solver.

            The ODE solver expect the features to be bi-dimensional, ie [batch, feature size].

            Args:
                times : ODE times, dimension [batch_size]
                flat_relative_coordinates :  features for every time step, dimension [batch_size, number of features].

            Returns:
                rhs: the right-hand-side of the corresponding ODE.
            """
            sigmas = self._get_exploding_variance_sigma(times)
            ode_prefactor = self._get_ode_prefactor(sigmas)

            relative_coordinates = einops.rearrange(flat_relative_coordinates,
                                                    "batch (natom space) -> batch natom space",
                                                    natom=self.number_of_atoms,
                                                    space=self.spatial_dimension)

            batch = {NOISY_RELATIVE_COORDINATES: map_relative_coordinates_to_unit_cell(relative_coordinates),
                     NOISE: sigmas.unsqueeze(-1),
                     TIME: times.unsqueeze(-1),
                     UNIT_CELL: unit_cell,
                     CARTESIAN_FORCES: torch.zeros_like(relative_coordinates)  # TODO: handle forces correctly.
                     }

            # Shape [batch_size, number of atoms, spatial dimension]
            sigma_normalized_scores = self.sigma_normalized_score_network(batch)
            flat_sigma_normalized_scores = einops.rearrange(sigma_normalized_scores,
                                                            "batch natom space -> batch (natom space)")

            return -ode_prefactor.unsqueeze(-1) * flat_sigma_normalized_scores

        return ode_term

    def sample(self, number_of_samples: int, device: torch.device, unit_cell: torch.Tensor) -> torch.Tensor:
        """Sample.

        This method draws a position sample.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.
            unit_cell: unit cell definition in Angstrom.
                Tensor of dimensions [number_of_samples, spatial_dimension, spatial_dimension]

        Returns:
            samples: relative coordinates samples.
        """
        ode_term = self.generate_ode_term(unit_cell)

        initial_relative_coordinates = (
            map_relative_coordinates_to_unit_cell(self.initialize(number_of_samples)).to(device))

        y0 = einops.rearrange(initial_relative_coordinates, 'batch natom space -> batch (natom space)')

        evaluation_times = torch.linspace(self.tf, self.t0, self.noise_parameters.total_time_steps).to(device)

        t_eval = einops.repeat(evaluation_times, 't -> batch t', batch=number_of_samples)

        term = to.ODETerm(ode_term)
        step_method = to.Dopri5(term=term)

        step_size_controller = to.IntegralController(atol=self.absolute_solver_tolerance,
                                                     rtol=self.relative_solver_tolerance,
                                                     term=term)
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        # jit_solver = torch.compile(solver) # Compilation is not necessary, and breaks on the cluster...
        jit_solver = solver

        logger.info("Starting ODE solver...")
        sol = jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))
        logger.info("ODE solver Finished.")

        if self.record_samples:
            self.record_sample(ode_term, sol, evaluation_times, unit_cell)

        # sol.ys has dimensions [number of samples, number of times, number of features]
        # only the final time (ie, t0) is the real sample.
        flat_relative_coordinates = sol.ys[:, -1, :]

        relative_coordinates = einops.rearrange(flat_relative_coordinates,
                                                'batch (natom space) -> batch natom space',
                                                natom=self.number_of_atoms,
                                                space=self.spatial_dimension)

        return map_relative_coordinates_to_unit_cell(relative_coordinates)

    def record_sample(self, ode_term: Callable, sol: Solution, evaluation_times: torch.Tensor, unit_cell: torch.Tensor):
        """Record sample.

        This method takes care of recomputing the normalized score on the solution trajectory and record it to the
        sample trajectory object.
        Args:
            ode_term : the Callable that is used to compute the rhs of the solved ODE.
            sol : the solution object obtained when solving the ODE.
            evaluation_times : times along the trajectory.
            unit_cell : unit cell definition in Angstrom.

        Returns:
            None
        """
        number_of_samples = sol.ys.shape[0]

        self.sample_trajectory_recorder.record_unit_cell(unit_cell)
        record_relative_coordinates = einops.rearrange(sol.ys,
                                                       'batch times (natom space) -> batch times natom space',
                                                       natom=self.number_of_atoms,
                                                       space=self.spatial_dimension)
        sigmas = self._get_exploding_variance_sigma(evaluation_times)
        ode_prefactor = self._get_ode_prefactor(sigmas)
        list_flat_normalized_scores = []
        for time_idx, (time, gamma) in enumerate(zip(evaluation_times, ode_prefactor)):
            times = time * torch.ones(number_of_samples).to(sol.ys)
            # The score network must be called again to get scores at intermediate times
            flat_normalized_score = -ode_term(times=times,
                                              flat_relative_coordinates=sol.ys[:, time_idx]) / gamma
            list_flat_normalized_scores.append(flat_normalized_score)
        record_normalized_scores = einops.rearrange(torch.stack(list_flat_normalized_scores),
                                                    "time batch (natom space) -> batch time natom space",
                                                    natom=self.number_of_atoms, space=self.spatial_dimension)
        self.sample_trajectory_recorder.record_ode_solution(times=evaluation_times,
                                                            sigmas=sigmas,
                                                            relative_coordinates=record_relative_coordinates,
                                                            normalized_scores=record_normalized_scores,
                                                            stats=sol.stats,
                                                            status=sol.status)

    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        relative_coordinates = torch.rand(number_of_samples, self.number_of_atoms, self.spatial_dimension)
        return relative_coordinates

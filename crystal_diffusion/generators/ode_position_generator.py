import logging
from typing import Callable

import einops
import torch
import torchode as to

from crystal_diffusion.generators.position_generator import PositionGenerator
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


class ExplodingVarianceODEPositionGenerator(PositionGenerator):
    """Exploding Variance ODE Position Generator.

    This class generates position samples by solving an ordinary differential equation (ODE).
    It assumes that the diffusion noise is parameterized in the 'Exploding Variance' scheme.
    """

    def __init__(self,
                 noise_parameters: NoiseParameters,
                 number_of_atoms: int,
                 spatial_dimension: int,
                 sigma_normalized_score_network: ScoreNetwork,
                 record_samples: bool = False,
                 ):
        """Init method.

        Args:
            noise_parameters : the diffusion noise parameters.
            number_of_atoms : the number of atoms to sample.
            spatial_dimension : the dimension of space.
            sigma_normalized_score_network : the score network to use for drawing samples.
            record_samples : should samples be recorded.
        """
        self.noise_parameters = noise_parameters
        assert self.noise_parameters.total_time_steps >= 2, \
            "There must at least be two time steps in the noise parameters to define the limits t0 and tf."
        self.number_of_atoms = number_of_atoms
        self.spatial_dimension = spatial_dimension

        self.sigma_normalized_score_network = sigma_normalized_score_network

        self.t0 = 0.0  # The "initial diffusion time", corresponding to the physical distribution.
        self.tf = 1.0  # The "final diffusion time", corresponding to the uniform distribution.

        self.record_samples = record_samples

        if record_samples:
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

        evaluation_times = torch.linspace(self.tf, self.t0, self.noise_parameters.total_time_steps)

        t_eval = einops.repeat(evaluation_times, 't -> batch t', batch=number_of_samples)

        term = to.ODETerm(ode_term)
        step_method = to.Dopri5(term=term)
        # TODO: parameterize the tolerances
        step_size_controller = to.IntegralController(atol=1e-3, rtol=1e-3, term=term)
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        jit_solver = torch.compile(solver)

        logger.info("Starting ODE solver...")
        sol = jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))
        logger.info("ODE solver Finished.")

        if self.record_samples:
            # Only do these operations if they are required!
            self.sample_trajectory_recorder.record_unit_cell(unit_cell)
            record_relative_coordinates = einops.rearrange(sol.ys,
                                                           'batch times (natom space) -> batch times natom space',
                                                           natom=self.number_of_atoms,
                                                           space=self.spatial_dimension)
            self.sample_trajectory_recorder.record_ode_solution(times=sol.ts,
                                                                relative_coordinates=record_relative_coordinates,
                                                                stats=sol.stats,
                                                                status=sol.status)

        # sol.ys has dimensions [number of samples, number of times, number of features]
        # only the final time (ie, t0) is the real sample.
        flat_relative_coordinates = sol.ys[:, -1, :]

        relative_coordinates = einops.rearrange(flat_relative_coordinates,
                                                'batch (natom space) -> batch natom space',
                                                natom=self.number_of_atoms,
                                                space=self.spatial_dimension)

        return map_relative_coordinates_to_unit_cell(relative_coordinates)

    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        relative_coordinates = torch.rand(number_of_samples, self.number_of_atoms, self.spatial_dimension)
        return relative_coordinates

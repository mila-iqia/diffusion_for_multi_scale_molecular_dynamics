import logging
from dataclasses import dataclass

import einops
import torch
import torchsde

from crystal_diffusion.generators.position_generator import (
    PositionGenerator, SamplingParameters)
from crystal_diffusion.models.score_networks import ScoreNetwork
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from crystal_diffusion.utils.sample_trajectory import SDESampleTrajectory

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SDESamplingParameters(SamplingParameters):
    """Hyper-parameters for diffusion sampling with the sde algorithm."""
    algorithm: str = 'sde'
    sde_type: str = 'ito'
    method: str = 'euler'
    adaptative: bool = False
    absolute_solver_tolerance: float = 1.0e-7  # the absolute error tolerance passed to the SDE solver.
    relative_solver_tolerance: float = 1.0e-5  # the relative error tolerance passed to the SDE solver.


class SDE(torch.nn.Module):
    """SDE.

    This class computes the drift and the diffusion coefficients in order to be consisent with the expectations
    of the torchsde library.
    """
    noise_type = 'diagonal'  # we assume that there is a distinct Wiener process for each component.
    sde_type = 'ito'

    def __init__(self,
                 noise_parameters: NoiseParameters,
                 sampling_parameters: SDESamplingParameters,
                 sigma_normalized_score_network: ScoreNetwork,
                 unit_cells: torch.Tensor,
                 initial_diffusion_time: torch.Tensor,
                 final_diffusion_time: torch.Tensor):
        """Init method.

        This class will provide drift and diffusion for the torchsde solver. The SDE will be solved
        "backwards" in diffusion time, such that the sde time will start where the diffusion ends, and vice-versa.

        Args:
            noise_parameters: parameters defining the noise schedule.
            sampling_parameters : parameters defining the sampling procedure.
            sigma_normalized_score_network : the score network to use for drawing samples.
            unit_cells: unit cell definition in Angstrom.
                Tensor of dimensions [number_of_samples, spatial_dimension, spatial_dimension]
            initial_diffusion_time : initial diffusion time. Dimensionless tensor.
            final_diffusion_time : final diffusion time. Dimensionless tensor.
        """
        super().__init__()
        self.sde_type = sampling_parameters.sde_type
        self.noise_parameters = noise_parameters
        self.sigma_normalized_score_network = sigma_normalized_score_network
        self.unit_cells = unit_cells
        self.number_of_atoms = sampling_parameters.number_of_atoms
        self.spatial_dimension = sampling_parameters.spatial_dimension
        self.initial_diffusion_time = initial_diffusion_time
        self.final_diffusion_time = final_diffusion_time

    def _get_exploding_variance_sigma(self, diffusion_time: torch.Tensor) -> torch.Tensor:
        """Get Exploding Variance Sigma.

        In the 'exploding variance' scheme, the noise is defined by

            sigma(t) = sigma_min^{1- t} x sigma_max^{t}

        Args:
            diffusion_time : diffusion time

        Returns:
            sigma: value of the noise parameter.
        """
        sigma = (self.noise_parameters.sigma_min ** (1.0 - diffusion_time)
                 * self.noise_parameters.sigma_max ** diffusion_time)
        return sigma

    def _get_diffusion_coefficient_g_squared(self, diffusion_time: torch.Tensor) -> torch.Tensor:
        """Get diffusion coefficient g(t)^2.

        The noise is given by sigma(t) = sigma_{min} (sigma_{max} / sigma_{min})^t
        and g(t)^2 = d/dt sigma(t)^2.

        Args:
            diffusion_time : Diffusion time.

        Returns:
            coefficient_g : the coefficient g(t)
        """
        s_min = torch.tensor(self.noise_parameters.sigma_min)
        ratio = torch.tensor(self.noise_parameters.sigma_max / self.noise_parameters.sigma_min)

        g_squared = 2.0 * (s_min * ratio ** diffusion_time) ** 2 * torch.log(ratio)
        return g_squared

    def _get_diffusion_time(self, sde_time: torch.Tensor) -> torch.Tensor:
        """Get diffusion time.

        Args:
            sde_time : SDE time.

        Returns:
            diffusion_time: the time for the diffusion process.
        """
        return self.final_diffusion_time - sde_time

    def f(self, sde_time: torch.Tensor, flat_relative_coordinates: torch.Tensor) -> torch.Tensor:
        """Drift function.

        Args:
            sde_time : time for the SDE. Dimensionless tensor.
            flat_relative_coordinates : time-dependent state. This corresponds to flattened atomic coordinates.
                Dimension: [batch_size (number of samples), natoms x spatial_dimension]

        Returns:
            f : the drift.
        """
        diffusion_time = self._get_diffusion_time(sde_time)

        sigma_normalized_scores = self.get_sigma_normalized_score(diffusion_time, flat_relative_coordinates)
        flat_sigma_normalized_scores = einops.rearrange(sigma_normalized_scores,
                                                        "batch natom space -> batch (natom space)")

        g_squared = self._get_diffusion_coefficient_g_squared(diffusion_time)
        sigma = self._get_exploding_variance_sigma(diffusion_time)
        # Careful! The prefactor must account for the following facts:
        #   -  the SDE time is NEGATIVE the diffusion time; this introduces a minus sign dt_{diff} = -dt_{sde}
        #   -  what our model calculates is the NORMALIZED score (ie, Score x sigma). We must thus divide by sigma.
        prefactor = g_squared / sigma

        return prefactor * flat_sigma_normalized_scores

    def get_sigma_normalized_score(self, diffusion_time: torch.Tensor,
                                   flat_relative_coordinates: torch.Tensor) -> torch.Tensor:
        """Get sigma normalized score.

        This is a utility method to wrap around the computation of the sigma normalized score in this context,
        dealing with dimensions and tensor reshapes as needed.

        Args:
            diffusion_time : the diffusion time. Dimensionless tensor.
            flat_relative_coordinates : the flat relative coordinates.
                Dimension [batch_size, natoms x spatial_dimensions]

        Returns:
            sigma_normalized_score: the sigma normalized score.
                Dimension [batch_size, natoms, spatial_dimensions]
        """
        batch_size = flat_relative_coordinates.shape[0]
        sigma = self._get_exploding_variance_sigma(diffusion_time)
        sigmas = einops.repeat(sigma.unsqueeze(0), "1 -> batch 1", batch=batch_size)
        times = einops.repeat(diffusion_time.unsqueeze(0), "1 -> batch 1", batch=batch_size)

        relative_coordinates = einops.rearrange(flat_relative_coordinates,
                                                "batch (natom space) -> batch natom space",
                                                natom=self.number_of_atoms,
                                                space=self.spatial_dimension)
        batch = {NOISY_RELATIVE_COORDINATES: map_relative_coordinates_to_unit_cell(relative_coordinates),
                 NOISE: sigmas,
                 TIME: times,
                 UNIT_CELL: self.unit_cells,
                 CARTESIAN_FORCES: torch.zeros_like(relative_coordinates)  # TODO: handle forces correctly.
                 }
        # Shape [batch_size, number of atoms, spatial dimension]
        sigma_normalized_scores = self.sigma_normalized_score_network(batch)
        return sigma_normalized_scores

    def g(self, sde_time, y):
        """Diffusion function."""
        diffusion_time = self._get_diffusion_time(sde_time)
        g_squared = self._get_diffusion_coefficient_g_squared(diffusion_time)
        g_of_t = torch.sqrt(g_squared)

        return g_of_t * torch.ones_like(y)


class ExplodingVarianceSDEPositionGenerator(PositionGenerator):
    """Exploding Variance SDE Position Generator.

    This class generates position samples by solving a stochastic differential equation (SDE).
    It assumes that the diffusion noise is parameterized in the 'Exploding Variance' scheme.
    """
    def __init__(self,
                 noise_parameters: NoiseParameters,
                 sampling_parameters: SDESamplingParameters,
                 sigma_normalized_score_network: ScoreNetwork,
                 ):
        """Init method.

        Args:
            noise_parameters : the diffusion noise parameters.
            sampling_parameters: the parameters needed for sampling.
            sigma_normalized_score_network : the score network to use for drawing samples.
        """
        self.initial_diffusion_time = torch.tensor(0.0)
        self.final_diffusion_time = torch.tensor(1.0)

        self.noise_parameters = noise_parameters
        self.sigma_normalized_score_network = sigma_normalized_score_network
        self.sampling_parameters = sampling_parameters

        self.number_of_atoms = sampling_parameters.number_of_atoms
        self.spatial_dimension = sampling_parameters.spatial_dimension
        self.absolute_solver_tolerance = sampling_parameters.absolute_solver_tolerance
        self.relative_solver_tolerance = sampling_parameters.relative_solver_tolerance
        self.record_samples = sampling_parameters.record_samples
        if self.record_samples:
            self.sample_trajectory_recorder = SDESampleTrajectory()

    def get_sde(self, unit_cells: torch.Tensor) -> SDE:
        """Get SDE."""
        return SDE(noise_parameters=self.noise_parameters,
                   sampling_parameters=self.sampling_parameters,
                   sigma_normalized_score_network=self.sigma_normalized_score_network,
                   unit_cells=unit_cells,
                   initial_diffusion_time=self.initial_diffusion_time,
                   final_diffusion_time=self.final_diffusion_time)

    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        relative_coordinates = torch.rand(number_of_samples, self.number_of_atoms, self.spatial_dimension)
        return relative_coordinates

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
        sde = self.get_sde(unit_cell)
        sde.to(device)

        initial_relative_coordinates = (
            map_relative_coordinates_to_unit_cell(self.initialize(number_of_samples)).to(device))
        y0 = einops.rearrange(initial_relative_coordinates, 'batch natom space -> batch (natom space)')

        sde_times = torch.linspace(self.initial_diffusion_time, self.final_diffusion_time,
                                   self.noise_parameters.total_time_steps).to(device)

        dt = (self.final_diffusion_time - self.initial_diffusion_time) / (self.noise_parameters.total_time_steps - 1)

        with torch.no_grad():
            # Dimensions [number of time steps, number of samples, natom x spatial_dimension]
            logger.info("Starting SDE solver...")
            ys = torchsde.sdeint(sde, y0, sde_times,
                                 method=self.sampling_parameters.method,
                                 dt=dt,
                                 adaptive=self.sampling_parameters.adaptative,
                                 atol=self.sampling_parameters.absolute_solver_tolerance,
                                 rtol=self.sampling_parameters.relative_solver_tolerance)
            logger.info("SDE solver Finished.")

        if self.record_samples:
            self.record_sample(sde, ys, sde_times)

        # only the final sde time (ie, diffusion time t0) is the real sample.
        flat_relative_coordinates = ys[-1, :, :]

        relative_coordinates = einops.rearrange(flat_relative_coordinates,
                                                'batch (natom space) -> batch natom space',
                                                natom=self.number_of_atoms,
                                                space=self.spatial_dimension)

        return map_relative_coordinates_to_unit_cell(relative_coordinates)

    def record_sample(self, sde: SDE, ys: torch.Tensor, sde_times: torch.Tensor):
        """Record sample.

        This  takes care of recomputing the normalized score on the solution trajectory and record it to the
        sample trajectory object.
        Args:
            sde: the SDE object that provides drift and diffusion.
            ys: the SDE solutions.
                Dimensions [number of time steps, number of samples, natom x spatial_dimension]
            sde_times : times along the sde trajectory.

        Returns:
            None
        """
        self.sample_trajectory_recorder.record_unit_cell(sde.unit_cells)

        list_normalized_scores = []
        sigmas = []
        evaluation_times = []
        # Reverse the times since the SDE times are inverted compared to the diffusion times.
        for sde_time, flat_relative_coordinates in zip(sde_times.flip(dims=(0,)), ys.flip(dims=(0,))):
            diffusion_time = sde._get_diffusion_time(sde_time)
            sigma = sde._get_exploding_variance_sigma(diffusion_time)
            sigmas.append(sigma)
            evaluation_times.append(diffusion_time)

            with torch.no_grad():
                normalized_scores = sde.get_sigma_normalized_score(diffusion_time, flat_relative_coordinates)
            list_normalized_scores.append(normalized_scores)

        sigmas = torch.tensor(sigmas)
        evaluation_times = torch.tensor(evaluation_times)

        record_normalized_scores = einops.rearrange(torch.stack(list_normalized_scores),
                                                    "time batch natom space  -> batch time natom space")

        record_relative_coordinates = einops.rearrange(ys.flip(dims=(0,)),
                                                       'times batch (natom space) -> batch times natom space',
                                                       natom=self.number_of_atoms,
                                                       space=self.spatial_dimension)

        self.sample_trajectory_recorder.record_sde_solution(times=evaluation_times,
                                                            sigmas=sigmas,
                                                            relative_coordinates=record_relative_coordinates,
                                                            normalized_scores=record_normalized_scores)

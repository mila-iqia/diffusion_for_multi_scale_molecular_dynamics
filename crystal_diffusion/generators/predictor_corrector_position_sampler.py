import logging
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from crystal_diffusion.models.score_networks.score_network import ScoreNetwork
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)
from crystal_diffusion.samplers.variance_sampler import (
    ExplodingVarianceSampler, NoiseParameters)
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from crystal_diffusion.utils.sample_trajectory import (NoOpSampleTrajectory,
                                                       SampleTrajectory)

logger = logging.getLogger(__name__)


class PredictorCorrectorPositionSampler(ABC):
    """This defines the interface for position samplers."""

    def __init__(self, number_of_discretization_steps: int, number_of_corrector_steps: int, spatial_dimension: int,
                 **kwargs):
        """Init method."""
        assert number_of_discretization_steps > 0, "The number of discretization steps should be larger than zero"
        assert number_of_corrector_steps >= 0, "The number of corrector steps should be non-negative"

        self.number_of_discretization_steps = number_of_discretization_steps
        self.number_of_corrector_steps = number_of_corrector_steps
        self.spatial_dimension = spatial_dimension

    def sample(self, number_of_samples: int, device: torch.device, unit_cell: torch.Tensor) -> torch.Tensor:
        """Sample.

        This method draws a sample using the PR sampler algorithm.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.
            unit_cell: unit cell definition in Angstrom.
                Tensor of dimensions [number_of_samples, spatial_dimension, spatial_dimension]

        Returns:
            samples: relative coordinates samples.
        """
        assert unit_cell.size() == (number_of_samples, self.spatial_dimension, self.spatial_dimension), \
            "Unit cell passed to sample should be of size (number of sample, spatial dimension, spatial dimension" \
            + f"Got {unit_cell.size()}"

        x_ip1 = map_relative_coordinates_to_unit_cell(self.initialize(number_of_samples)).to(device)
        forces = torch.zeros_like(x_ip1)

        for i in tqdm(range(self.number_of_discretization_steps - 1, -1, -1)):
            x_i = map_relative_coordinates_to_unit_cell(self.predictor_step(x_ip1, i + 1, unit_cell, forces))
            for _ in range(self.number_of_corrector_steps):
                x_i = map_relative_coordinates_to_unit_cell(self.corrector_step(x_i, i, unit_cell, forces))
            x_ip1 = x_i
        return x_i

    @abstractmethod
    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        pass

    @abstractmethod
    def predictor_step(self, x_ip1: torch.Tensor, ip1: int, unit_cell: torch.Tensor, cartesian_forces: torch.Tensor
                       ) -> torch.Tensor:
        """Predictor step.

        It is assumed that there are N predictor steps, with index "i" running from N-1 to 0.

        Args:
            x_ip1 : sampled relative coordinates at step "i + 1".
            ip1 : index "i + 1"
            unit_cell: sampled unit cell at time step "i + 1".
            cartesian_forces: forces conditioning the diffusion process

        Returns:
            x_i : sampled relative coordinates after the predictor step.
        """
        pass

    @abstractmethod
    def corrector_step(self, x_i: torch.Tensor, i: int, unit_cell: torch.Tensor, cartesian_forces: torch.Tensor
                       ) -> torch.Tensor:
        """Corrector step.

        It is assumed that there are N predictor steps, with index "i" running from N-1 to 0.
        For each value of "i", there are M corrector steps.
        Args:
            x_i : sampled relative coordinates at step "i".
            i : index "i" OF THE PREDICTOR STEP.
            unit_cell: sampled unit cell at time step i.
            cartesian_forces: forces conditioning the diffusion process

        Returns:
            x_i_out : sampled relative coordinates after the corrector step.
        """
        pass


class AnnealedLangevinDynamicsSampler(PredictorCorrectorPositionSampler):
    """Annealed Langevin Dynamics Sampler.

    This class implements the annealed Langevin Dynamics sampling of
    Song & Ermon 2019, namely:
        "Generative Modeling by Estimating Gradients of the Data Distribution"
    """

    def __init__(self,
                 noise_parameters: NoiseParameters,
                 number_of_corrector_steps: int,
                 number_of_atoms: int,
                 spatial_dimension: int,
                 sigma_normalized_score_network: ScoreNetwork,
                 record_samples: bool = False,
                 positions_require_grad: bool = False
                 ):
        """Init method."""
        super().__init__(number_of_discretization_steps=noise_parameters.total_time_steps,
                         number_of_corrector_steps=number_of_corrector_steps,
                         spatial_dimension=spatial_dimension)
        self.noise_parameters = noise_parameters
        self.positions_require_grad = positions_require_grad
        sampler = ExplodingVarianceSampler(noise_parameters)
        self.noise, self.langevin_dynamics = sampler.get_all_sampling_parameters()
        self.number_of_atoms = number_of_atoms
        self.sigma_normalized_score_network = sigma_normalized_score_network

        if record_samples:
            self.sample_trajectory_recorder = SampleTrajectory()
        else:
            self.sample_trajectory_recorder = NoOpSampleTrajectory()

    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        relative_coordinates = torch.rand(number_of_samples, self.number_of_atoms, self.spatial_dimension)
        if self.positions_require_grad:
            relative_coordinates.requires_grad_(True)
        return relative_coordinates

    def _draw_gaussian_sample(self, number_of_samples):
        return torch.randn(number_of_samples, self.number_of_atoms, self.spatial_dimension)

    def _get_sigma_normalized_scores(self, x: torch.Tensor, time: float,
                                     noise: float, unit_cell: torch.Tensor, cartesian_forces: torch.Tensor
                                     ) -> torch.Tensor:
        """Get sigma normalized scores.

        Args:
            x : relative coordinates, of shape [number_of_samples, number_of_atoms, spatial_dimension]
            time : time at which to evaluate the score
            noise: the diffusion sigma parameter corresponding to the time at which to evaluate the score
            unit_cell: unit cell definition in Angstrom of shape [number_of_samples, spatial_dimension,
                spatial_dimension]
            cartesian_forces: forces to condition the sampling from. Shape [number_of_samples, number_of_atoms,
                spatial_dimension]

        Returns:
            sigma normalized score: sigma x Score(x, t).
        """
        number_of_samples = x.shape[0]

        time_tensor = time * torch.ones(number_of_samples, 1).to(x)
        noise_tensor = noise * torch.ones(number_of_samples, 1).to(x)
        augmented_batch = {NOISY_RELATIVE_COORDINATES: x, TIME: time_tensor, NOISE: noise_tensor, UNIT_CELL: unit_cell,
                           CARTESIAN_FORCES: cartesian_forces}

        # TODO do not hard-code conditional to False - need to be able to condition sampling
        predicted_normalized_scores = self.sigma_normalized_score_network(augmented_batch, conditional=False)
        return predicted_normalized_scores

    def predictor_step(self, x_i: torch.Tensor, index_i: int, unit_cell: torch.Tensor, cartesian_forces: torch.Tensor
                       ) -> torch.Tensor:
        """Predictor step.

        Args:
            x_i : sampled relative coordinates, at time step i.
            index_i : index of the time step.
            unit_cell: sampled unit cell at time step i.
            cartesian_forces: forces conditioning the sampling process

        Returns:
            x_im1 : sampled relative coordinates, at time step i - 1.
        """
        assert 1 <= index_i <= self.number_of_discretization_steps, \
            "The predictor step can only be invoked for index_i between 1 and the total number of discretization steps."

        number_of_samples = x_i.shape[0]
        z = self._draw_gaussian_sample(number_of_samples).to(x_i)

        idx = index_i - 1  # python starts indices at zero
        t_i = self.noise.time[idx].to(x_i)
        g_i = self.noise.g[idx].to(x_i)
        g2_i = self.noise.g_squared[idx].to(x_i)
        sigma_i = self.noise.sigma[idx].to(x_i)
        sigma_score_i = self._get_sigma_normalized_scores(x_i, t_i, sigma_i, unit_cell, cartesian_forces)
        x_im1 = x_i + g2_i / sigma_i * sigma_score_i + g_i * z

        self.sample_trajectory_recorder.record_unit_cell(unit_cell=unit_cell)
        self.sample_trajectory_recorder.record_predictor_step(i_index=index_i, time=t_i, sigma=sigma_i,
                                                              x_i=x_i, x_im1=x_im1, scores=sigma_score_i)

        return x_im1

    def corrector_step(self, x_i: torch.Tensor, index_i: int, unit_cell: torch.Tensor, cartesian_forces: torch.Tensor
                       ) -> torch.Tensor:
        """Corrector Step.

        Args:
            x_i : sampled relative coordinates, at time step i.
            index_i : index of the time step.
            unit_cell: sampled unit cell at time step i.
            cartesian_forces: forces conditioning the sampling

        Returns:
            corrected x_i : sampled relative coordinates, after corrector step.
        """
        assert 0 <= index_i <= self.number_of_discretization_steps - 1, \
            ("The corrector step can only be invoked for index_i between 0 and "
             "the total number of discretization steps minus 1.")

        number_of_samples = x_i.shape[0]
        z = self._draw_gaussian_sample(number_of_samples).to(x_i)

        # The Langevin dynamics array are indexed with [0,..., N-1]
        eps_i = self.langevin_dynamics.epsilon[index_i].to(x_i)
        sqrt_2eps_i = self.langevin_dynamics.sqrt_2_epsilon[index_i].to(x_i)

        if index_i == 0:
            # TODO: we are extrapolating here; the score network will never have seen this time step...
            sigma_i = self.noise_parameters.sigma_min  # no need to change device, this is a float
            t_i = 0.  # same for device - this is a float
        else:
            idx = index_i - 1  # python starts indices at zero
            sigma_i = self.noise.sigma[idx].to(x_i)
            t_i = self.noise.time[idx].to(x_i)

        sigma_score_i = self._get_sigma_normalized_scores(x_i, t_i, sigma_i, unit_cell, cartesian_forces)

        corrected_x_i = x_i + eps_i / sigma_i * sigma_score_i + sqrt_2eps_i * z

        self.sample_trajectory_recorder.record_corrector_step(i_index=index_i, time=t_i,
                                                              sigma=sigma_i, x_i=x_i, corrected_x_i=corrected_x_i,
                                                              scores=sigma_score_i)

        return corrected_x_i

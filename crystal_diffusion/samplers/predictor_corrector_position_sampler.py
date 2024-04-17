import logging
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from crystal_diffusion.models.score_network import ScoreNetwork
from crystal_diffusion.samplers.noisy_position_sampler import \
    map_positions_to_unit_cell
from crystal_diffusion.samplers.variance_sampler import (
    ExplodingVarianceSampler, NoiseParameters)

logger = logging.getLogger(__name__)


class PredictorCorrectorPositionSampler(ABC):
    """This defines the interface for position samplers."""

    def __init__(self, number_of_discretization_steps: int, number_of_corrector_steps: int, **kwargs):
        """Init method."""
        assert number_of_discretization_steps > 0, "The number of discretization steps should be larger than zero"
        assert number_of_corrector_steps >= 0, "The number of corrector steps should be non-negative"

        self.number_of_discretization_steps = number_of_discretization_steps
        self.number_of_corrector_steps = number_of_corrector_steps

    def sample(self, number_of_samples: int) -> torch.Tensor:
        """Sample.

        This method draws a sample using the PR sampler algorithm.

        Args:
            number_of_samples : number of samples to draw.

        Returns:
            position samples: position samples.
        """
        x_ip1 = map_positions_to_unit_cell(self.initialize(number_of_samples))
        logger.info("Starting position sampling")
        for i in tqdm(range(self.number_of_discretization_steps - 1, -1, -1)):
            x_i = map_positions_to_unit_cell(self.predictor_step(x_ip1, i + 1))
            for j in range(self.number_of_corrector_steps):
                x_i = map_positions_to_unit_cell(self.corrector_step(x_i, i))
            x_ip1 = x_i

        return x_i

    @abstractmethod
    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        pass

    @abstractmethod
    def predictor_step(self, x_ip1: torch.Tensor, ip1: int) -> torch.Tensor:
        """Predictor step.

        It is assumed that there are N predictor steps, with index "i" running from N-1 to 0.

        Args:
            x_ip1 : sampled relative positions at step "i + 1".
            ip1 : index "i + 1"

        Returns:
            x_i : sampled relative positions after the predictor step.
        """
        pass

    @abstractmethod
    def corrector_step(self, x_i: torch.Tensor, i: int) -> torch.Tensor:
        """Corrector step.

        It is assumed that there are N predictor steps, with index "i" running from N-1 to 0.
        For each value of "i", there are M corrector steps.
        Args:
            x_i : sampled relative positions at step "i".
            i : index "i" OF THE PREDICTOR STEP.

        Returns:
            x_i_out : sampled relative positions after the corrector step.
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
                 ):
        """Init method."""
        super().__init__(number_of_discretization_steps=noise_parameters.total_time_steps,
                         number_of_corrector_steps=number_of_corrector_steps)
        self.noise_parameters = noise_parameters
        sampler = ExplodingVarianceSampler(noise_parameters)
        self.noise, self.langevin_dynamics = sampler.get_all_sampling_parameters()
        self.number_of_atoms = number_of_atoms
        self.spatial_dimension = spatial_dimension
        self.sigma_normalized_score_network = sigma_normalized_score_network

    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        return torch.rand(number_of_samples, self.number_of_atoms, self.spatial_dimension)

    def _draw_gaussian_sample(self, number_of_samples):
        return torch.randn(number_of_samples, self.number_of_atoms, self.spatial_dimension)

    def _get_sigma_normalized_scores(self, x: torch.Tensor, time: float) -> torch.Tensor:
        """Get sigma normalized scores.

        Args:
            x : relative positions, of shape [number_of_samples, number_of_atoms, spatial_dimension]
            time : time at which to evaluate the score

        Returns:
            sigma normalized score: sigma x Score(x, t).
        """
        pos_key = self.sigma_normalized_score_network.position_key
        time_key = self.sigma_normalized_score_network.timestep_key

        number_of_samples = x.shape[0]

        time_tensor = time * torch.ones(number_of_samples, 1)
        augmented_batch = {pos_key: x, time_key: time_tensor}
        with torch.no_grad():
            predicted_normalized_scores = self.sigma_normalized_score_network(augmented_batch)

        return predicted_normalized_scores

    def predictor_step(self, x_i: torch.Tensor, index_i: int) -> torch.Tensor:
        """Predictor step.

        Args:
            x_i : sampled relative positions, at time step i.
            index_i : index of the time step.

        Returns:
            x_im1 : sampled relative positions, at time step i - 1.
        """
        assert 1 <= index_i <= self.number_of_discretization_steps, \
            "The predictor step can only be invoked for index_i between 1 and the total number of discretization steps."

        number_of_samples = x_i.shape[0]
        z = self._draw_gaussian_sample(number_of_samples)

        idx = index_i - 1  # python starts indices at zero
        t_i = self.noise.time[idx]
        g_i = self.noise.g[idx]
        g2_i = self.noise.g_squared[idx]
        sigma_i = self.noise.sigma[idx]

        sigma_score_i = self._get_sigma_normalized_scores(x_i, t_i)

        x_im1 = x_i + g2_i / sigma_i * sigma_score_i + g_i * z

        return x_im1

    def corrector_step(self, x_i: torch.Tensor, index_i: int) -> torch.Tensor:
        """Corrector Step.

        Args:
            x_i : sampled relative positions, at time step i.
            index_i : index of the time step.

        Returns:
            corrected x_i : sampled relative positions, after corrector step.
        """
        assert 0 <= index_i <= self.number_of_discretization_steps - 1, \
            ("The corrector step can only be invoked for index_i between 0 and "
             "the total number of discretization steps minus 1.")

        number_of_samples = x_i.shape[0]
        z = self._draw_gaussian_sample(number_of_samples)

        # The Langevin dynamics array are indexed with [0,..., N-1]
        eps_i = self.langevin_dynamics.epsilon[index_i]
        sqrt_2eps_i = self.langevin_dynamics.sqrt_2_epsilon[index_i]

        if index_i == 0:
            # TODO: we are extrapolating here; the score network will never have seen this time step...
            sigma_i = self.noise_parameters.sigma_min
            t_i = 0.
        else:
            idx = index_i - 1  # python starts indices at zero
            sigma_i = self.noise.sigma[idx]
            t_i = self.noise.time[idx]

        sigma_score_i = self._get_sigma_normalized_scores(x_i, t_i)

        corrected_x_i = x_i + eps_i / sigma_i * sigma_score_i + sqrt_2eps_i * z

        return corrected_x_i

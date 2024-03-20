from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple

import torch

Noise = namedtuple("Noise", ["time", "sigma", "sigma_squared", "g", "g_squared"])


@dataclass
class NoiseParameters:
    """Variance parameters."""

    total_time_steps: int
    # Default values come from the paper:
    #   "Torsional Diffusion for Molecular Conformer Generation",
    # The original values in the paper are
    #   sigma_min = 0.01 pi , sigma_σmax = pi
    # However, they consider angles from 0 to 2pi as their coordinates:
    # here we divide by 2pi because our space is in the range [0, 1).
    sigma_min: float = 0.005
    sigma_max: float = 0.5


class ExplodingVarianceSampler:
    """Exploding Variance Sampler.

    This class is responsible for creating the all the quantities
    needed for noise generation.

    This implementation will use "exponential diffusion" as discussed in
    the paper "Torsional Diffusion for Molecular Conformer Generation".
    """

    def __init__(self, noise_parameters: NoiseParameters):
        """Init method.

        Args:
            noise_parameters: parameters that define the noise schedule.
        """
        self.noise_parameters = noise_parameters
        self._time_array = torch.linspace(0, 1, noise_parameters.total_time_steps)

        self._sigma_array = self._create_sigma_array(noise_parameters, self._time_array)
        self._sigma_squared_array = self._sigma_array**2

        self._g_squared_array = self._create_g_squared_array(self._sigma_squared_array)
        self._g_array = torch.sqrt(self._g_squared_array)

        self._maximum_random_index = noise_parameters.total_time_steps - 1
        self._minimum_random_index = 1  # we don't want to randomly sample "0".

    @staticmethod
    def _create_sigma_array(
        noise_parameters: NoiseParameters, time_array: torch.Tensor
    ) -> torch.Tensor:
        sigma_min = noise_parameters.sigma_min
        sigma_max = noise_parameters.sigma_max

        sigma = sigma_min ** (1.0 - time_array) * sigma_max**time_array
        return sigma

    @staticmethod
    def _create_g_squared_array(sigma_squared_array: torch.Tensor) -> torch.Tensor:
        nan_tensor = torch.tensor([float("nan")])
        return torch.cat(
            [nan_tensor, sigma_squared_array[1:] - sigma_squared_array[:-1]]
        )

    def _get_random_time_step_indices(self, shape: Tuple[int]) -> torch.Tensor:
        """Random time step indices.

        Generate random indices that correspond to valid time steps.
        This sampling avoids index "0", which corresponds to time "0".

        Args:
            shape: shape of the random index array.

        Returns:
            time_step_indices: random time step indices in a tensor of shape "shape".
        """
        random_indices = torch.randint(
            self._minimum_random_index,
            self._maximum_random_index
            + 1,  # +1 because the maximum value is not sampled
            size=shape,
        )
        return random_indices

    def get_random_noise_sample(self, batch_size: int) -> Noise:
        """Get random noise sample.

        It is assumed that a batch is of the form [batch_size, (dimensions of a configuration)].
        In order to train a diffusion model, a configuration must be "noised" to a time t with a parameter sigma(t).
        Different values can be used for different configurations: correspondingly, this method returns
        one random time per element in the batch.


        Args:
            batch_size : number of configurations in a batch,

        Returns:
            noise_sample: a collection of all the noise parameters (t, sigma, sigma^2, g, g^2)
                for some random indices. All the arrays are of dimension [batch_size].
        """
        indices = self._get_random_time_step_indices((batch_size,))
        times = self._time_array.take(indices)
        sigmas = self._sigma_array.take(indices)
        sigmas_squared = self._sigma_squared_array.take(indices)
        gs = self._g_array.take(indices)
        gs_squared = self._g_squared_array.take(indices)

        return Noise(
            time=times,
            sigma=sigmas,
            sigma_squared=sigmas_squared,
            g=gs,
            g_squared=gs_squared,
        )

    def get_all_noise(self) -> Noise:
        """Get all noise.

        All the internal noise parameter arrays, passed as a Noise object.

        Returns:
            all_noise: a collection of all the noise parameters (t, sigma, sigma^2, g, g^2)
                for all indices. The arrays are all of dimension [total_time_steps].
        """
        return Noise(
            time=self._time_array,
            sigma=self._sigma_array,
            sigma_squared=self._sigma_squared_array,
            g=self._g_array,
            g_squared=self._g_squared_array,
        )

from dataclasses import dataclass

import torch

from crystal_diffusion.samplers.time_sampler import TimeSampler


@dataclass
class VarianceParameters:
    """Variance parameters."""
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

    This class is responsible for creating the variances
    needed for noise generation.

    This implementation will use "exponential diffusion" as discussed in
    the paper "Torsional Diffusion for Molecular Conformer Generation".
    """

    def __init__(
        self, variance_parameters: VarianceParameters, time_sampler: TimeSampler
    ):
        """Init method.

        Args:
            variance_parameters: parameters that define the variance schedule.
            time_sampler: object that can sample time steps.
        """
        self._sigma_square_array = self._create_sigma_square_array(
            variance_parameters, time_sampler
        )
        self._g_square_array = self._create_g_square_array(self._sigma_square_array)

        self._maximum_index = len(self._sigma_square_array) - 1

    def _create_sigma_square_array(
        self, variance_parameters: VarianceParameters, time_sampler: TimeSampler
    ) -> torch.Tensor:

        t = time_sampler.time_step_array

        sigma_min = variance_parameters.sigma_min
        sigma_max = variance_parameters.sigma_max

        sigma = sigma_min**(1.0 - t) * sigma_max**t
        return sigma**2

    def _create_g_square_array(self, sigma_square_array: torch.Tensor) -> torch.Tensor:
        nan_tensor = torch.tensor([float('nan')])
        return torch.cat([nan_tensor, sigma_square_array[1:] - sigma_square_array[:-1]])

    def get_variances(self, indices: torch.Tensor) -> torch.Tensor:
        """Get variances.

        Args:
            indices : indices to be extracted.

        Returns:
            variances: the variances at the specified indices.
        """
        assert torch.all(indices >= 0), "indices must be non-negative"
        assert torch.all(indices <= self._maximum_index), \
            f"indices must be smaller than or equal to {self._maximum_index}"
        return self._sigma_square_array.take(indices)

    def get_g_squared_factors(self, indices: torch.Tensor) -> torch.Tensor:
        """Get g squared factors.

        The g squared factors are defined as:

            g(t)^2 = sigma(t)^2 - sigma(t-1)^2

        Note that this is ill-defined at t=0.

        Args:
            indices: indices at which to get the g squared factors.

        Returns:
            g_square_factors : g squared factors at specified indices
        """
        assert torch.all(indices > 0), "g squared factor is ill defined at index zero."
        assert torch.all(indices <= self._maximum_index), \
            f"indices must be smaller than or equal to {self._maximum_index}"
        return self._g_square_array.take(indices)

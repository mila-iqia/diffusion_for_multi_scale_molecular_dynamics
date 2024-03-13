"""Noisy Position Sampler.

This module is responsible for sampling relative positions from the perturbation kernel.
"""
from typing import Tuple

import torch


class NoisyPositionSampler:
    """Noisy Position Sampler.

    This class provides methods to generate noisy positions, given real positions and
    a sigma parameter.

    The random samples are produced by a separate method to make this code easy to test.
    """
    @staticmethod
    def _get_gaussian_noise(shape: Tuple[int]) -> torch.Tensor:
        """Get Gaussian noise.

        Get a sample from N(0, 1) of dimensions shape.

        Args:
            shape : the shape of the sample.

        Returns:
            gaussian_noise: a sample from N(0, 1) of dimensions shape.
        """
        return torch.randn(shape)

    @staticmethod
    def get_noisy_position_sample(real_relative_positions: torch.Tensor, sigma: float) -> torch.Tensor:
        """Get noisy positions sample.

        This method draws a sample from the perturbation kernel centered on the real_relative_positions
        and with a variance parameter sigma. The sample is brought back into the periodic unit cell.

        Args:
            real_relative_positions : relative coordinates of real data. Should be between 0 and 1.
                relative_positions is assumed to have an arbitrary shape.
            sigma : variance of the perturbation kernel.

        Returns:
            noisy_relative_positions: a sample of noised relative positions, of the same shape as relative_positions.
        """
        noise = sigma * NoisyPositionSampler._get_gaussian_noise(real_relative_positions.shape)
        noisy_relative_positions = torch.remainder(real_relative_positions + noise, 1.0)
        return noisy_relative_positions

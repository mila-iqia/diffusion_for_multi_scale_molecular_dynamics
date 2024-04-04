"""Noisy Position Sampler.

This module is responsible for sampling relative positions from the perturbation kernel.
"""
from typing import Tuple

import torch


def map_positions_to_unit_cell(positions: torch.Tensor) -> torch.Tensor:
    """Map positions back to unit cell.

    The function torch.remainder does not always bring back the positions in the range [0, 1).
    If the input is very small and negative, torch.remainder returns 1. This is problematic
    when using floats instead of doubles.

    This method makes sure that the positions are mapped back in the [0, 1) range and does reasonable
    things when the position is close to the edge.

    See issues:
        https://github.com/pytorch/pytorch/issues/37743
        https://github.com/pytorch/pytorch/issues/24861

    Args:
        positions : atomic positions, tensor of arbitrary shape.

    Returns:
        relative_positions: atomic positions in the unit cell, ie, in the range [0, 1).
    """
    relative_positions = torch.remainder(positions, 1.0)
    relative_positions[relative_positions == 1.] = 0.
    return relative_positions


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
    def get_noisy_position_sample(real_relative_positions: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        """Get noisy positions sample.

        This method draws a sample from the perturbation kernel centered on the real_relative_positions
        and with a variance parameter sigma. The sample is brought back into the periodic unit cell.

        Note that sigmas is assumed to be of the same shape as real_relative_positions. There is no
        check that the sigmas are "all the same" for a given batch index: it is the user's responsibility to
        provide a consistent sigma, if the desired behavior is to noise a batch of configurations consistently.


        Args:
            real_relative_positions : relative coordinates of real data. Should be between 0 and 1.
                relative_positions is assumed to have an arbitrary shape.
            sigmas : variance of the perturbation kernel. Tensor is assumed to be of the same shape as
                real_relative_positions.

        Returns:
            noisy_relative_positions: a sample of noised relative positions, of the same shape as relative_positions.
        """
        assert real_relative_positions.shape == sigmas.shape, \
            "sigmas array is expected to be of the same shape as the real_relative_positions array"

        z_scores = NoisyPositionSampler._get_gaussian_noise(real_relative_positions.shape)
        noise = (sigmas * z_scores).to(real_relative_positions.device)
        noisy_relative_positions = map_positions_to_unit_cell(real_relative_positions + noise)
        return noisy_relative_positions

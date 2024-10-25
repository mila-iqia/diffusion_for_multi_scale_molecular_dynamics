"""Noisy Relative Coordinates.

This module is responsible for sampling relative coordinates from the perturbation kernel.
"""

from typing import Tuple

import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell


class NoisyRelativeCoordinates:
    """Noisy Relative Coordinates.

    This class provides methods to generate noisy relative coordinates, given real relative coordinates and
    a sigma parameter.
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
    def get_noisy_relative_coordinates_sample(
        real_relative_coordinates: torch.Tensor, sigmas: torch.Tensor
    ) -> torch.Tensor:
        """Get noisy relative coordinates sample.

        This method draws a sample from the perturbation kernel centered on the real_relative_coordinates
        and with a variance parameter sigma. The sample is brought back into the periodic unit cell.

        Note that sigmas is assumed to be of the same shape as real_relative_coordinates. There is no
        check that the sigmas are "all the same" for a given batch index: it is the user's responsibility to
        provide a consistent sigma, if the desired behavior is to noise a batch of configurations consistently.


        Args:
            real_relative_coordinates : relative coordinates of real data. Should be between 0 and 1.
                real_relative_coordinates is assumed to have an arbitrary shape.
            sigmas : variance of the perturbation kernel. Tensor is assumed to be of the same shape as
                real_relative_coordinates.

        Returns:
            noisy_relative_coordinates: a sample of noised relative coordinates, of the same
                shape as real_relative_coordinates.
        """
        assert (
            real_relative_coordinates.shape == sigmas.shape
        ), "sigmas array is expected to be of the same shape as the real_relative_coordinates array"

        z_scores = NoisyRelativeCoordinates._get_gaussian_noise(
            real_relative_coordinates.shape
        ).to(sigmas)
        noise = (sigmas * z_scores).to(real_relative_coordinates)
        noisy_relative_coordinates = map_relative_coordinates_to_unit_cell(
            real_relative_coordinates + noise
        )
        return noisy_relative_coordinates

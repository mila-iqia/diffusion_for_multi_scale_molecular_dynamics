"""Noisy Sampler.

This module is responsible for sampling relative positions from the perturbation kernel and the noisy atom types from
a noised distribution.
"""

from typing import Tuple

import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    q_xt_bar_xo


class NoisyRelativeCoordinatesSampler:
    """Noisy Relative Coordinates Sampler.

    This class provides methods to generate noisy relative coordinates, given real relative coordinates and
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

        z_scores = NoisyRelativeCoordinatesSampler._get_gaussian_noise(
            real_relative_coordinates.shape
        ).to(sigmas)
        noise = (sigmas * z_scores).to(real_relative_coordinates)
        noisy_relative_coordinates = map_relative_coordinates_to_unit_cell(
            real_relative_coordinates + noise
        )
        return noisy_relative_coordinates


class NoisyAtomTypesSampler:
    """Noisy Relative Coordinates Sampler.

    This class provides methods to generate noisy relative coordinates, given real relative coordinates and
    a sigma parameter.

    The random samples are produced by a separate method to make this code easy to test.
    """
    @staticmethod
    def _get_uniform_noise(shape: Tuple[int]) -> torch.Tensor:
        """Get uniform noise.

        Get a sample from U(0, 1) of dimensions shape.

        Args:
            shape : the shape of the sample.

        Returns:
            gaussian_noise: a sample from U(0, 1) of dimensions shape.
        """
        return torch.rand(shape)

    @staticmethod
    def get_noisy_atom_types_sample(
            real_onehot_atom_types: torch.Tensor, q_bar: torch.Tensor
    ) -> torch.Tensor:
        """Get noisy atom types sample.

        This method generates a sample using the transition probabilities defined by the q_bar matrices.

        Args:
            real_onehot_atom_types : atom types of the real sample. Assumed to be a one-hot vector. The size is assumed
                to be (..., num_classes + 1) where num_classes is the number of atoms.
            q_bar : cumulative transition matrices i.e. the q_bar in q(a_t | a_0) = a_0 \bar{Q}_t. Assumed to be of size
                (..., num_classes + 1, num_classes + 1)

        Returns:
            noisy_atom_types: a sample of noised atom types as classes, not 1-hot, of the same shape as
            real_onehot_atom_types except for the last dimension that is removed.
        """
        assert (
                real_onehot_atom_types.shape == q_bar.shape[:-1]
        ), "q_bar array first dimensions should match real_atom_types array"

        u_scores = NoisyAtomTypesSampler._get_uniform_noise(
            real_onehot_atom_types.shape
        ).to(q_bar)
        # we need to sample from q(x_t | x_0)
        posterior_xt = q_xt_bar_xo(real_onehot_atom_types, q_bar)
        # gumbel trick to sample from a distribution
        noise = -torch.log(-torch.log(u_scores)).to(real_onehot_atom_types.device)
        noisy_atom_types = torch.log(posterior_xt) + noise
        noisy_atom_types = torch.argmax(noisy_atom_types, dim=-1)
        return noisy_atom_types


class NoisyLatticeSampler:
    """Get noisy lattice vectors.

    This class provides methods to generate noisy relative coordinates, given the real vectors from data samples and
    a beta noise parameter.

    The random samples are produced by a separate method to make this code easy to test.

    TODO this is a placeholder
    """
    @staticmethod
    def get_noisy_lattice_vectors(real_lattice_vectors: torch.Tensor) -> torch.Tensor:
        """Get noisy lattice vectors.

        TODO this is a placeholder

        Args:
            real_lattice_vectors: lattice vectors from the sampled data

        Returns:
            real_lattice_vectors: a sample of noised lattice vectors. Placeholder for now.
        """
        return real_lattice_vectors

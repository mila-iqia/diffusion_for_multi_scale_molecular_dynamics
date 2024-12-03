from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(kw_only=True)
class LatticeDataParameters:
    """Information on the mean of the lattice parameters used for noising a sample.

    TODO: this might belong elsewhere
    """
    spatial_dimension: int = 3


class LatticeNoiser:
    """Lattice noiser.

    This class provides methods to generate noisy lattices.
    TODO this is a placeholder
    """
    def __init__(self, lattice_parameters: LatticeDataParameters):
        self.spatial_dimension = lattice_parameters.spatial_dimension

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

    def get_noisy_lattice_vectors(
        self,
        real_lattice_parameters: torch.Tensor,
        sigmas_n: torch.Tensor,
    ) -> torch.Tensor:
        r"""Get noisy lattice vectors.

        We consider the lattice parameters as a tensor with 6, 3, or 1 parameters for 3D, 2D, 1D.


        Args:
            real_lattice_parameters: lattice parameters from the sampled data. These parameters are not the lattice
                vector, but an array of dimension [spatial_dimension * (spatial_dimension + 1) / 2] containing the size
                of the orthogonal box and the angles.  # TODO review statement about angles
            sigmas_n: variance of the perturbation kernel rescaled by the number of atoms. Tensor is assumed to be of
                the same shape as real_lattice_parameters.

        Returns:
            noisy_lattice_parameters: a sample of noised lattice parameters as tensor of size
            [spatial_dimension * (spatial_dimension + 1) / 2].
        """
        assert (
            real_lattice_parameters.shape == sigmas_n.shape
        ), "sigmas array is expected to be of the same shape as the real_lattice_parameters array"

        z_scores = self._get_gaussian_noise(real_lattice_parameters.shape).to(sigmas_n)

        # we are not limiting the range of value to anything i.e. negative lattice parameters are allowed in the noisy
        # space
        noisy_lattice_parameters = sigmas_n * z_scores + real_lattice_parameters

        return noisy_lattice_parameters

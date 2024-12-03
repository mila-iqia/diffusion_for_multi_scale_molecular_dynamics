from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(kw_only=True)
class LatticeDataParameters:
    """Information on the mean of the lattice parameters used for noising a sample.

    TODO: this might belong elsewhere
    """
    inverse_average_density: float  # inverse of the average volume of unit cell scales by the number of atoms
    spatial_dimension: int = 3


class LatticeNoiser:
    """Lattice noiser.

    This class provides methods to generate noisy lattices.
    TODO this is a placeholder
    """
    def __init__(self, lattice_parameters: LatticeDataParameters):
        self.inverse_density = lattice_parameters.inverse_average_density
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
        alpha_bars: torch.Tensor,
        num_atoms: torch.Tensor,
    ) -> torch.Tensor:
        r"""Get noisy lattice vectors.

        We consider the lattice parameters as a tensor of spatial_dimension degrees of freedom i.e. we assume the unit
        cell is orthogonal.

        .. math::

            q(L_t | L_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}L_0 +
                (1 - \sqrt{\bar{\alpha}_t})\mu(n)I, (1-\sqrt{\bar{\alpha}_t})\sigma^2_t(n)I\right)

        Args:
            real_lattice_parameters: lattice parameters from the sampled data. This is not the lattice vector, but an
                array of dimension [spatial_dimension] containing the size of the orthogonal box. TODO add angles
            sigmas_n: variance of the perturbation kernel rescaled by the number of atoms. Tensor is assumed to be of
                the same shape as real_lattice_parameters.
            alpha_bars: cumulative noise scale. Tensor is assumed to be of the same shape as real_lattice_parameters.
            num_atoms: number of atoms in each sample. Tensor should be a 1D tensor matching the first dimension of the
                real_lattice_parameters tensor.

        Returns:
            noisy_lattice_parameters: a sample of noised lattice vectors as tensor of size [spatial_dimension].
        """
        # TODO add angles
        assert (
                real_lattice_parameters.shape == sigmas_n.shape
        ), "sigmas array is expected to be of the same shape as the real_lattice_parameters array"

        assert (
                alpha_bars.shape == sigmas_n.shape
        ), "sigmas array is expected to be of the same shape as the alpha_bars array"

        z_scores = LatticeNoiser._get_gaussian_noise(
            real_lattice_parameters.shape
        ).to(sigmas_n)

        noise_width = (1 - alpha_bars) * sigmas_n ** 2
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        average_density = torch.pow(num_atoms / self.inverse_density, 1 / self.spatial_dimension)
        noisy_lattice_parameters_avg = sqrt_alpha_bars * real_lattice_parameters + \
                                       (1 - sqrt_alpha_bars) * average_density
        noisy_lattice_parameters = noise_width * z_scores + noisy_lattice_parameters_avg

        return noisy_lattice_parameters

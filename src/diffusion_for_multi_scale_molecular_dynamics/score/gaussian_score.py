"""Gaussian Score.

This module implements the Gaussian score, which corresponds to the "perturbation kernel" for diffusion on the lattice
parameters.

The Gaussian takes the form

    K(l, l0) ~ exp[- |l - l0|^2 / 2 sigma^2]

The score is defined as S = nabla_l ln K(l, l0).
"""

import torch


def get_lattice_sigma_normalized_score(
    noisy_l: torch.Tensor, real_l: torch.Tensor, sigma_n: torch.Tensor, alpha_bar: torch.Tensor,
    num_atoms: torch.Tensor, inverse_density: float, spatial_dimension: int,
) -> torch.Tensor:
    r"""Get the sigma normalized score for lattice parameters.

    We compute this from a normal (non-wrapped) gaussian kernel.  Because we are using a variance-preserving approach
    for the lattice parameters, the width of the gaussian kernel is rescaled by the noise parameter :math:`alpha_bar`.

    .. math::

        \sigma_\textrm{eff}^2 = (1 - \bar{\alpha}) \sigma^2(n)

    with :math:`\sigma(n) = \frac{\sigma}{\sqrt[1/d]{n}}` with :math:`d` the number of spatial dimensions.

    Args:
        noisy_l : the noised lattice parameters used to evaluate the Gaussian score.
        real_l: the real (non-noised) lattice parameters to evaluate the Gaussian score.
        sigma_n : the variance in the definition of the variance-exploding diffusion - scaled by the number of atoms.
        alpha_bar : the cumulative variance in the definition of the variance-preserving diffusion.
        num_atoms: number of atoms in each element of the batch.
        inverse_density: average of the inverse density used to weight the identity component of the noised lattice
           parameters.
        spatial_dimension: number of spatial dimension.  # TODO might remove when dealing with angles properly


    Returns:
        sigma_score : the value of the sigma normalized score.
    """
    # rescale sigma by 1 - alpha_bar
    sigma_effective = torch.sqrt(1 - alpha_bar) * sigma_n

    average_density = torch.zeros_like(real_l)
    # compute :math:`\mu(n)`
    average_density[:, :spatial_dimension] = torch.pow(
        num_atoms[:, :spatial_dimension],
        1 / spatial_dimension,
    ) / inverse_density  # TODO add angles

    # we can get the sigma normalized score using a method similar to the brute force calculation in the wrapped
    # gaussian kernel implementation. We do not need a sum over k as the kernel is not periodic. This simplifies to the
    # -x / sigma
    # note that the lattice score rescales L_0 (the real lattice parameters) by a factor :math:`\sqrt{\bar{\alpha}}`.
    sigma_score = -(noisy_l - torch.sqrt(alpha_bar) * real_l - (1 - torch.sqrt(alpha_bar)) * average_density)
    sigma_score /= sigma_effective
    return sigma_score

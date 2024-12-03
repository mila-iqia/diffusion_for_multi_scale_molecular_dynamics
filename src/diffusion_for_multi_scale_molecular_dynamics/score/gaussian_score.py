"""Gaussian Score.

This module implements the Gaussian score, which corresponds to the "perturbation kernel" for diffusion on the lattice
parameters.

The Gaussian takes the form

    K(l, l0) ~ exp[- |l - l0|^2 / 2 sigma^2]

The score is defined as S = nabla_l ln K(l, l0).
"""
import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics.score.wrapped_gaussian_score import \
    get_coordinates_sigma_normalized_score


def get_lattice_sigma_normalized_score(
    delta_l: torch.Tensor, sigma_n: torch.Tensor, alpha_bar: torch.Tensor
) -> torch.Tensor:
    r"""Get the sigma normalized score for lattice parameters.

    We compute this from a normal (non-wrapped) gaussian kernel, which we can get by calling a wrapped gaussian kernel
    with k_max = 0 (which effectively removes the sum). Because we are using a variance-preserving approach for the
    lattice parameters, the width of the gaussian kernel is rescaled by the noise parameter :math:`alpha_bar`.

    .. math::

        \sigma_\textrm{eff}^2 = (1 - \bar{\alpha}) \sigma^2(n)

    with :math:`\sigma(n) = \frac{\sigma}{\sqrt[1/d]{n}}` with :math:`d` the number of spatial dimensions.

    Args:
        delta_l : the lattice parameters at which the wrapped Gaussian is evaluated.
        sigma_n : the variance in the definition of the variance-exploding diffusion - scaled by the number of atoms.
        alpha_bar : the cumulative variance in the definition of the variance-preserving diffusion.

    Returns:
        sigma_normalized_score : the value of the sigma normalized score.
    """
    # rescale sigma by 1 - alpha_bar
    sigma_effective = np.sqrt(1 - alpha_bar) * sigma_n
    # we can get the sigma normalized score using the wrapped gaussian method with k_max = 0 i.e. no sum
    # and using the effective sigma computed
    sigma_normalized_score = get_coordinates_sigma_normalized_score(
        delta_l,
        sigma_effective,
        kmax=0
    )
    return sigma_normalized_score

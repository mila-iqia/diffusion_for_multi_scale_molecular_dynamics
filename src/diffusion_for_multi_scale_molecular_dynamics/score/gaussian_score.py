"""Gaussian Score.

This module implements the Gaussian score, which corresponds to the "perturbation kernel" for diffusion on the lattice
parameters.

The Gaussian takes the form

    K(l, l0) ~ exp[- |l - l0|^2 / 2 sigma^2]

The score is defined as S = nabla_l ln K(l, l0).
"""

import torch


def get_lattice_sigma_normalized_score(
    noisy_l: torch.Tensor,
    real_l: torch.Tensor,
    sigma_n: torch.Tensor,
) -> torch.Tensor:
    r"""Get the sigma normalized score for lattice parameters.

    We compute this from a normal (non-wrapped) gaussian kernel.

    Args:
        noisy_l : the noised lattice parameters used to evaluate the Gaussian score.
        real_l: the real (non-noised) lattice parameters to evaluate the Gaussian score.
        sigma_n : the variance in the definition of the variance-exploding diffusion - scaled by the number of atoms.

    Returns:
        sigma_score : the value of the sigma normalized score.
    """
    # we can get the sigma normalized score using a method similar to the brute force calculation in the wrapped
    # gaussian kernel implementation. We do not need a sum over k as the kernel is not periodic. This simplifies to the
    # -x / sigma
    sigma_score = -(noisy_l - real_l) / sigma_n
    return sigma_score

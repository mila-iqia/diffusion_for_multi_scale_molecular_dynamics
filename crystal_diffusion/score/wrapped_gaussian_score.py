"""Wrapped Gaussian Score.

This module implements the "wrapped Gaussian" score, which corresponds to the "perturbation kernel" on a torus.

The wrapped Gaussian takes the form

    K(x, x0) ~ sum_{k in Z} exp[- |x - x0 + k|^2 / 2 sigma^2], x, x0 in [0, 1).

The score is defined as S = nabla_x ln K(x, x0). Implementing this naively yields an expression that
potentially converges slowly when sigma is large.

The code below will implement expressions that leverages the "Ewald trick" (ie, part of the sum in real space,
part of the sum in Fourier space) to insure quick convergence for any sigma. The formula for small
sigma is derived to avoid division by a number that is very close to zero, or summing very large terms that
can overlflow.

Also, what is computed is actually "sigma2 x S" (ie, the "sigma normalized score"). This is because
S ~ 1/ sigma^2; since sigma can be small, this makes the raw score arbitrarily large, and it is better to
manipulate numbers of small magnitude.

Relevant papers:
    "Torsional Diffusion for Molecular Conformer Generation",
        Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, Tommi Jaakkola

    "Riemannian Score-Based Generative Modelling", Bortoli et al.
"""
from typing import Optional

import numpy as np
import torch

SIGMA_THRESHOLD = 1.0 / np.sqrt(2.0 * np.pi)
U_THRESHOLD = 0.5


def get_expected_sigma_normalized_score_brute_force(u: float, sigma: float, kmax: Optional[int] = None) -> float:
    """Brute force implementation.

    A brute force implementation of the sigma normalized score to check that the main code is correct.
    This is only useful if summed to convergence, which is expensive for large sigma!

    Args:
        u : the relative position at which the wrapped Gaussian is evaluated. Assumed between 0 and 1.
        sigma : the variance in the definition of the wrapped Gaussian.
        kmax : if provided, the sum will be from -kmax to kmax. If not provided, a large
            default value will be used.

    Returns:
        sigma_normalized_score : the value of the normalized score.
    """
    z = 0.0
    sigma2_derivative_z = 0.0

    if kmax is None:
        kmax = np.max([1, np.round(10 * sigma)])

    for k in np.arange(-kmax, kmax + 1):
        upk = u + k
        exp = np.exp(-0.5 * upk**2 / sigma**2)

        z += exp
        sigma2_derivative_z += -upk * exp

    return sigma2_derivative_z / z


def get_sigma_normalized_score(
    relative_positions: torch.Tensor, sigmas: torch.Tensor, kmax: int
) -> torch.Tensor:
    """Get the sigma normalized score.

    This method branches to different formulas depending on the values of sigma and relative position
    to insures rapid convergence and numerical stability.

    Args:
        relative_positions : input relative coordinates: should be between 0 and 1.
            relative_positions is assumed to have an arbitrary shape.
        sigmas : the values of sigma. Should have the same dimension as relative positions.
        kmax : largest positive integer in the sum. The sum is from -kmax to +kmax.

    Returns:
        list_sigma_normalized_score : the sigma_normalized_scores, in the same shape as
            relative_positions.
    """
    assert kmax >= 0, "kmax must be a non negative integer"
    assert (sigmas > 0).all(), "All values of sigma should be larger than zero."
    assert torch.logical_and(
        relative_positions >= 0, relative_positions < 1
    ).all(), "the relative positions should all be in [0, 1)"
    assert (
        sigmas.shape == relative_positions.shape
    ), "The relative_positions and sigmas inputs should have the same shape"

    total_number_of_elements = relative_positions.nelement()
    list_u = relative_positions.view(total_number_of_elements)
    list_sigma = sigmas.view(total_number_of_elements)

    # The dimension of list_k is [2 kmax + 1].
    list_k = torch.arange(-kmax, kmax + 1)

    # Initialize a results array, and view it as a flat list.
    # Since "flat_view" is a view on "sigma_normalized_scores",  sigma_normalized_scores is updated
    # when we assign values in flat_view (both tensors share the same underlying data structure).
    sigma_normalized_scores = torch.zeros_like(relative_positions)
    flat_view = sigma_normalized_scores.view(total_number_of_elements)

    mask_calculators = [
        _get_small_sigma_small_u_mask,
        _get_small_sigma_large_u_mask,
        _get_large_sigma_mask,
    ]
    score_calculators = [
        _get_sigma_normalized_s1a,
        _get_sigma_normalized_s1b,
        _get_sigma_normalized_s2,
    ]

    for mask_calculator, score_calculator in zip(mask_calculators, score_calculators):
        mask = mask_calculator(list_u, list_sigma)
        if mask.any():
            flat_view[mask] = score_calculator(list_u[mask], list_sigma[mask], list_k)

    return sigma_normalized_scores


def _get_small_sigma_small_u_mask(list_u: torch.Tensor, list_sigma: torch.Tensor) -> torch.Tensor:
    """Get the boolean mask for small sigma and small u.

    Args:
        list_u : the relative positions, with shape [Nu].
        list_sigma : the values of sigma, one value for each value of u, with shape [Nu].

    Returns:
        mask_1a : an array of booleans of shape [Nu]
    """
    return torch.logical_and(list_sigma <= SIGMA_THRESHOLD, list_u < U_THRESHOLD)


def _get_small_sigma_large_u_mask(list_u: torch.Tensor, list_sigma: torch.Tensor) -> torch.Tensor:
    """Get the boolean mask for small sigma and large u.

    Args:
        list_u : the relative positions, with shape [Nu].
        list_sigma : the values of sigma, one value for each value of u, with shape [Nu].

    Returns:
        mask_1b : an array of booleans of shape [Nu]
    """
    return torch.logical_and(list_sigma <= SIGMA_THRESHOLD, list_u >= U_THRESHOLD)


def _get_large_sigma_mask(list_u: torch.Tensor, list_sigma: torch.Tensor) -> torch.Tensor:
    """Get the boolean mask for large sigma.

    Args:
        list_u : NOT USED. Only passed for compatibility with the other mask calculators.
        list_sigma : the values of sigma, with shape [N].

    Returns:
        mask_2 : an array of booleans of shape [N]
    """
    return list_sigma > SIGMA_THRESHOLD


def _get_s1a_exponential(
    list_u: torch.Tensor, list_sigma: torch.Tensor, list_k: torch.Tensor
) -> torch.Tensor:
    """Get the exponential terms for small sigma and 0 <= u < 0.5.

    Args:
        list_u : the relative positions, with shape [Nu].
        list_sigma : the values of sigma, one value for each value of u, with shape [Nu].
        list_k : the integer values that will be summed over, with shape [Nk].

    Returns:
        exponential : the exponential terms, with shape [Nu, Nk].
    """
    # Broadcast to shape [Nu, Nk]
    column_u = list_u.view(list_u.nelement(), 1)
    column_sigma = list_sigma.view(list_u.nelement(), 1)
    exponential = torch.exp(
        -0.5 * (list_k**2 + 2.0 * column_u * list_k) / column_sigma**2
    )
    return exponential


def _get_s1b_exponential(
    list_u: torch.Tensor, list_sigma: torch.Tensor, list_k: torch.Tensor
) -> torch.Tensor:
    """Get the exponential terms for small sigma and 0.5 <= u < 1.

    Args:
        list_u : the relative positions, with shape [Nu].
        list_sigma : the values of sigma, one value for each value of u, with shape [Nu].
        list_k : the integer values that will be summed over, with shape [Nk].

    Returns:
        exponential : the exponential terms, with shape [Nu, Nk].
    """
    # Broadcast to shape [Nu, Nk]
    # Broadcast to shape [Nu, Nk]
    column_u = list_u.view(list_u.nelement(), 1)
    column_sigma = list_sigma.view(list_u.nelement(), 1)

    exponential = torch.exp(
        -0.5
        * ((list_k**2 - 1.0) + 2.0 * column_u * (list_k + 1.0))
        / column_sigma**2
    )
    return exponential


def _get_sigma_normalized_s1_from_exponential(
    exponential: torch.Tensor, list_u: torch.Tensor, list_k: torch.Tensor
) -> torch.Tensor:
    """Get one of the contributions to the S1 score, assuming the exponentials has been computed and is passed as input.

    Args:
        exponential : the exponential terms, with shape [Nu, Nk].
        list_u : the relative positions, with shape [Nu].
        list_k : the integer values that will be summed over, with shape [Nk].

    Returns:
        sigma_normalized_score_component : the corresponding sigma score, with shape [Nu]
    """
    # Sum is on Nk
    column_u = list_u.view(list_u.nelement(), 1)
    numerator = ((torch.ones_like(column_u) * list_k) * exponential).sum(dim=1)
    denominator = exponential.sum(dim=1)

    list_sigma_normalized_score = -list_u - (numerator / denominator)
    return list_sigma_normalized_score


def _get_sigma_normalized_s1a(
    list_u: torch.Tensor, list_sigma: torch.Tensor, list_k: torch.Tensor
) -> torch.Tensor:
    """Get the sigma normalized score for small sigma and 0 <= u < 0.5.

    This method assumes that the inputs are appropriate.

    Args:
        list_u : the relative positions, with shape [Nu].
        list_sigma : the values of sigma, one value for each value of u, with shape [Nu].
        list_k : the integer values that will be summed over, with shape [Nk].

    Returns:
        list_normalized_score : the sigma^2 x s1a scores, with shape [Nu].
    """
    exponential = _get_s1a_exponential(list_u, list_sigma, list_k)
    return _get_sigma_normalized_s1_from_exponential(exponential, list_u, list_k)


def _get_sigma_normalized_s1b(
    list_u: torch.Tensor, list_sigma: torch.Tensor, list_k: torch.Tensor
) -> torch.Tensor:
    """Get the sigma normalized score for small sigma and 0.5 <= u < 1.

    This method assumes that the inputs are appropriate.

    Args:
        list_u : the relative positions, with shape [Nu].
        list_sigma : the values of sigma, one value for each value of u, with shape [Nu].
        list_k : the integer values that will be summed over, with shape [Nk].

    Returns:
        list_column_normalized_score : the sigma^2 x s1b scores, with shape [Nu].
    """
    exponential = _get_s1b_exponential(list_u, list_sigma, list_k)
    return _get_sigma_normalized_s1_from_exponential(exponential, list_u, list_k)


def _get_sigma_normalized_s2(
    list_u: torch.Tensor, list_sigma: torch.Tensor, list_k: torch.Tensor
) -> torch.Tensor:
    """Get the sigma normalized score for large sigma.

    This method assumes that the inputs are appropriate.

    Args:
        list_u : the relative positions, with shape [Nu].
        list_sigma : the values of sigma, one value for each value of u, with shape [Nu].
        list_k : the integer values that will be summed over, with shape [Nk].

    Returns:
        list_normalized_score : the sigma^2 x s2 scores, with shape [Nu].
    """
    column_u = list_u.view(list_u.nelement(), 1)
    column_sigma = list_sigma.view(list_u.nelement(), 1)

    # Broadcast the various components to shape [Nu, Nk]
    upk = column_u + list_k
    gu = column_u * list_k
    sigma_g = column_sigma * list_k
    g = torch.ones_like(column_u) * list_k
    sig = column_sigma * torch.ones_like(list_k)

    exp_upk = (-np.pi * upk**2).exp()
    exp_sigma_g = (-2.0 * np.pi**2 * sigma_g**2).exp()
    exp_g = (-np.pi * g**2).exp()

    g_exponential_combination = np.sqrt(2.0 * np.pi) * sig * exp_sigma_g - exp_g

    cos = torch.cos(2.0 * np.pi * gu)
    sin = torch.sin(2.0 * np.pi * gu)

    # The sum is over Nk, leaving arrays of dimensions [Nu]
    z2 = exp_upk.sum(dim=1) + (g_exponential_combination * cos).sum(dim=1)
    deriv_z2 = -2.0 * np.pi * ((upk * exp_upk).sum(dim=1) + (g * g_exponential_combination * sin).sum(dim=1))
    list_sigma_normalized_scores_s2 = list_sigma**2 * deriv_z2 / z2

    return list_sigma_normalized_scores_s2
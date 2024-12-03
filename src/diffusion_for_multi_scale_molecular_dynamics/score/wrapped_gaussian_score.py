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

Also, what is computed is actually "sigma x S" (ie, the "sigma normalized score"). This is because, as argued
in section 4.2 of the paper
    "Generative Modeling by Estimating Gradients of the Data Distribution", Song & Ermon
at convergence we expect |S| ~ 1 / sigma. Normalizing the score should lead to numbers of the same order of magnitude.

Relevant papers:
    "Torsional Diffusion for Molecular Conformer Generation",
        Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, Tommi Jaakkola

    "Riemannian Score-Based Generative Modelling", Bortoli et al.

    "Generative Modeling by Estimating Gradients of the Data Distribution", Song & Ermon
"""

from typing import Optional

import einops
import numpy as np
import torch

SIGMA_THRESHOLD = torch.Tensor([1.0 / np.sqrt(2.0 * np.pi)])
U_THRESHOLD = torch.Tensor([0.5])


def get_log_wrapped_gaussians(
    relative_coordinates: torch.tensor, sigmas: torch.tensor, kmax: int
):
    """Get Log Wrapped Gaussians.

    Args:
        relative_coordinates : input relative coordinates: should be between 0 and 1.
            relative_coordinates has dimensions [..., number_of_atoms, spatial_dimension], where (...)
            are arbitrary batch dimensions.
        sigmas : the values of sigma. Should have the same dimension as relative coordinates.
        kmax : largest positive integer in the sum. The sum is from -kmax to +kmax.

    Returns:
        log_wrapped_gaussians: log of wrapped gaussian values, of dimensions [...], namely the batch dimensions.
    """
    device = relative_coordinates.device
    assert (
        sigmas.device == device
    ), "relative_coordinates and sigmas should be on the same device."

    assert (
        relative_coordinates.shape == sigmas.shape
    ), "The relative coordinates and sigmas array should have the same shape."

    assert (
        len(relative_coordinates.shape) >= 3
    ), "relative_coordinates should have at least 3 dimensions."

    input_shape = relative_coordinates.shape

    # The dimension of list_k is [2 kmax + 1].
    list_k = torch.arange(-kmax, kmax + 1).to(device)

    # Broadcast to shape [Nu, 1]
    column_u = einops.rearrange(relative_coordinates, "... -> (...) 1")
    column_sigma = einops.rearrange(sigmas, "... -> (...) 1")

    norm = torch.tensor(2 * torch.pi).sqrt() * column_sigma.squeeze(-1)

    flat_log_norm = torch.log(norm)

    # Broadcast to shape [Nu, Nk]
    exponentials = -0.5 * (column_u + list_k) ** 2 / column_sigma**2

    flat_logsumexp = torch.logsumexp(exponentials, dim=-1)

    flat_log_gaussians = flat_logsumexp - flat_log_norm

    log_gaussians = flat_log_gaussians.reshape(input_shape)

    log_wrapped_gaussians = log_gaussians.sum(dim=[-2, -1])
    return log_wrapped_gaussians


def get_sigma_normalized_score_brute_force(
    u: float, sigma: float, kmax: Optional[int] = None
) -> float:
    """Brute force implementation.

    A brute force implementation of the sigma normalized score to check that the main code is correct.
    This is only useful if summed to convergence, which is expensive for large sigma!

    Args:
        u : the relative coordinates at which the wrapped Gaussian is evaluated. Assumed between 0 and 1.
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

    sigma2_score = sigma2_derivative_z / z
    sigma_score = sigma2_score / sigma

    return sigma_score


def get_coordinates_sigma_normalized_score(
    relative_coordinates: torch.Tensor, sigmas: torch.Tensor, kmax: int
) -> torch.Tensor:
    """Get the sigma normalized score for relative coordinates from the wrapped gaussian kernel.

    This method branches to different formulas depending on the values of sigma and relative coordinates
    to insures rapid convergence and numerical stability.

    Args:
        relative_coordinates : input relative coordinates: should be between 0 and 1.
            relative_coordinates is assumed to have an arbitrary shape.
        sigmas : the values of sigma. Should have the same dimension as relative coordinates.
        kmax : largest positive integer in the sum. The sum is from -kmax to +kmax.

    Returns:
        list_sigma_normalized_score : the sigma_normalized_scores, in the same shape as
            relative_coordinates.
    """
    assert kmax >= 0, "kmax must be a non negative integer"
    assert (sigmas > 0).all(), "All values of sigma should be larger than zero."
    assert torch.logical_and(
        relative_coordinates >= 0, relative_coordinates < 1
    ).all(), "the relative coordinates should all be in [0, 1)"
    assert (
        sigmas.shape == relative_coordinates.shape
    ), "The relative_coordinates and sigmas inputs should have the same shape"

    device = relative_coordinates.device
    assert (
        sigmas.device == device
    ), "relative_coordinates and sigmas should be on the same device."

    total_number_of_elements = relative_coordinates.nelement()
    list_u = relative_coordinates.view(total_number_of_elements)
    list_sigma = sigmas.reshape(total_number_of_elements)

    # The dimension of list_k is [2 kmax + 1].
    list_k = torch.arange(-kmax, kmax + 1).to(device)

    # Initialize a results array, and view it as a flat list.
    # Since "flat_view" is a view on "sigma_normalized_scores",  sigma_normalized_scores is updated
    # when we assign values in flat_view (both tensors share the same underlying data structure).
    sigma_normalized_scores = torch.zeros_like(relative_coordinates)
    flat_view = sigma_normalized_scores.view(total_number_of_elements)

    mask_calculators = [
        _get_small_sigma_small_u_mask,
        _get_small_sigma_large_u_mask,
        _get_large_sigma_mask,
    ]
    score_calculators = [
        _get_sigma_normalized_score_1a,
        _get_sigma_normalized_score_1b,
        _get_sigma_normalized_s2,
    ]

    for mask_calculator, score_calculator in zip(mask_calculators, score_calculators):
        mask = mask_calculator(list_u, list_sigma)
        if mask.any():
            flat_view[mask] = score_calculator(list_u[mask], list_sigma[mask], list_k)

    return sigma_normalized_scores


def _get_small_sigma_small_u_mask(
    list_u: torch.Tensor, list_sigma: torch.Tensor
) -> torch.Tensor:
    """Get the boolean mask for small sigma and small u.

    Args:
        list_u : the relative coordinates, with shape [Nu].
        list_sigma : the values of sigma, one value for each value of u, with shape [Nu].

    Returns:
        mask_1a : an array of booleans of shape [Nu]
    """
    device = list_u.device
    return torch.logical_and(
        list_sigma.to(device) <= SIGMA_THRESHOLD.to(device),
        list_u < U_THRESHOLD.to(device),
    )


def _get_small_sigma_large_u_mask(
    list_u: torch.Tensor, list_sigma: torch.Tensor
) -> torch.Tensor:
    """Get the boolean mask for small sigma and large u.

    Args:
        list_u : the relative coordinates, with shape [Nu].
        list_sigma : the values of sigma, one value for each value of u, with shape [Nu].

    Returns:
        mask_1b : an array of booleans of shape [Nu]
    """
    device = list_u.device
    return torch.logical_and(
        list_sigma.to(device) <= SIGMA_THRESHOLD.to(device),
        list_u >= U_THRESHOLD.to(device),
    )


def _get_large_sigma_mask(
    list_u: torch.Tensor, list_sigma: torch.Tensor
) -> torch.Tensor:
    """Get the boolean mask for large sigma.

    Args:
        list_u : NOT USED. Only passed for compatibility with the other mask calculators.
        list_sigma : the values of sigma, with shape [N].

    Returns:
        mask_2 : an array of booleans of shape [N]
    """
    device = list_u.device
    return list_sigma.to(device) > SIGMA_THRESHOLD.to(device)


def _get_s1a_exponential(
    list_u: torch.Tensor, list_sigma: torch.Tensor, list_k: torch.Tensor
) -> torch.Tensor:
    """Get the exponential terms for small sigma and 0 <= u < 0.5.

    Args:
        list_u : the relative coordinates, with shape [Nu].
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
        list_u : the relative coordinates, with shape [Nu].
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
        -0.5 * ((list_k**2 - 1.0) + 2.0 * column_u * (list_k + 1.0)) / column_sigma**2
    )
    return exponential


def _get_sigma_square_times_score_1_from_exponential(
    exponential: torch.Tensor, list_u: torch.Tensor, list_k: torch.Tensor
) -> torch.Tensor:
    """Get one of the contributions to the S1 score, assuming the exponentials has been computed and is passed as input.

    Args:
        exponential : the exponential terms, with shape [Nu, Nk].
        list_u : the relative coordinates, with shape [Nu].
        list_k : the integer values that will be summed over, with shape [Nk].

    Returns:
        sigma_square_times_score_component : the corresponding score multiplied by sigma^2, with shape [Nu].
    """
    # Sum is on Nk
    column_u = list_u.view(list_u.nelement(), 1)
    numerator = ((torch.ones_like(column_u) * list_k) * exponential).sum(dim=1)
    denominator = exponential.sum(dim=1)

    list_sigma_square_times_score = -list_u - (numerator / denominator)
    return list_sigma_square_times_score


def _get_sigma_normalized_score_1a(
    list_u: torch.Tensor, list_sigma: torch.Tensor, list_k: torch.Tensor
) -> torch.Tensor:
    """Get the sigma times the score for small sigma and 0 <= u < 0.5.

    This method assumes that the inputs are appropriate.

    Args:
        list_u : the relative coordinates, with shape [Nu].
        list_sigma : the values of sigma, one value for each value of u, with shape [Nu].
        list_k : the integer values that will be summed over, with shape [Nk].

    Returns:
        list_sigma_normalized_score : the sigma x s1a scores, with shape [Nu].
    """
    exponential = _get_s1a_exponential(list_u, list_sigma, list_k)
    list_sigma_square_times_score = _get_sigma_square_times_score_1_from_exponential(
        exponential, list_u, list_k
    )
    list_normalized_score = list_sigma_square_times_score / list_sigma
    return list_normalized_score


def _get_sigma_normalized_score_1b(
    list_u: torch.Tensor, list_sigma: torch.Tensor, list_k: torch.Tensor
) -> torch.Tensor:
    """Get the sigma times the score for small sigma and 0.5 <= u < 1.

    This method assumes that the inputs are appropriate.

    Args:
        list_u : the relative coordinates, with shape [Nu].
        list_sigma : the values of sigma, one value for each value of u, with shape [Nu].
        list_k : the integer values that will be summed over, with shape [Nk].

    Returns:
        list_sigma_normalized_score : the sigma x s1b scores, with shape [Nu].
    """
    exponential = _get_s1b_exponential(list_u, list_sigma, list_k)
    list_sigma_square_times_score = _get_sigma_square_times_score_1_from_exponential(
        exponential, list_u, list_k
    )
    list_normalized_score = list_sigma_square_times_score / list_sigma
    return list_normalized_score


def _get_sigma_normalized_s2(
    list_u: torch.Tensor, list_sigma: torch.Tensor, list_k: torch.Tensor
) -> torch.Tensor:
    """Get the sigma normalized score for large sigma.

    This method assumes that the inputs are appropriate.

    Args:
        list_u : the relative coordinates, with shape [Nu].
        list_sigma : the values of sigma, one value for each value of u, with shape [Nu].
        list_k : the integer values that will be summed over, with shape [Nk].

    Returns:
        list_normalized_score : the sigma x s2 scores, with shape [Nu].
    """
    numerical_type = list_u.dtype

    column_u = list_u.view(list_u.nelement(), 1)
    column_sigma = list_sigma.view(list_u.nelement(), 1)

    # Broadcast the various components to shape [Nu, Nk]
    upk = column_u + list_k
    gu = column_u * list_k
    sigma_g = column_sigma * list_k
    g = torch.ones_like(column_u) * list_k
    sig = column_sigma * torch.ones_like(list_k)

    pi = torch.tensor(np.pi, dtype=numerical_type)

    exp_upk = (-pi * upk**2).exp()
    exp_sigma_g = (-2.0 * pi**2 * sigma_g**2).exp()
    exp_g = (-pi * g**2).exp()

    g_exponential_combination = torch.sqrt(2.0 * pi) * sig * exp_sigma_g - exp_g

    cos = torch.cos(2.0 * pi * gu)
    sin = torch.sin(2.0 * pi * gu)

    # The sum is over Nk, leaving arrays of dimensions [Nu]
    z2 = exp_upk.sum(dim=1) + (g_exponential_combination * cos).sum(dim=1)
    deriv_z2 = (
        -2.0
        * pi
        * (
            (upk * exp_upk).sum(dim=1)
            + (g * g_exponential_combination * sin).sum(dim=1)
        )
    )
    list_sigma_normalized_scores_s2 = list_sigma * deriv_z2 / z2

    return list_sigma_normalized_scores_s2

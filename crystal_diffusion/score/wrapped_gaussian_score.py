"""Wrapped Gaussian Score.

This module implements the "wrapped Gaussian" score, which corresponds to the "perturbation kernel" on a torus.

The wrapped Gaussian takes the form

    K(x, x0) ~ sum_{k in Z} exp[- |x - x0 + k|^2 / 2 sigma^2], x, x0 in [0, 1).

The score is defined as S = nabla_x ln K(x, x0). Implementing this naively yields an expression that
potentially converges slowly when sigma is large.

The code below will implement expressions that leverages the "Ewald trick" (ie, part of the sum in real space,
part of the sum in Fourier space) to insures quick convergence for any sigma. Also, the formula for small
sigma is derived to avoid division by a number that is very close to zero.

Also, what will be computed will actually be "sigma2 x S" (ie, the "sigma normalized score"). This is because
S ~ 1/ sigma^2; since sigma can be small, this makes the raw score arbitrarily large, and it is better to
manipulate numbers of small magnitude.

Relevant papers:
    "Torsional Diffusion for Molecular Conformer Generation",
        Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, Tommi Jaakkola

    "Riemannian Score-Based Generative Modelling", Bortoli et al.
"""
import numpy as np
import torch

SIGMA_THRESHOLD = 1. / np.sqrt(2. * np.pi)


def get_sigma_normalized_score_small_sigma(relative_positions: torch.Tensor, sigma: float, kmax: int):
    """Get the sigma normalized score for the "small sigma" implementation.

    Args:
        relative_positions (torch.Tensor): input relative coordinates: should be between 0 and 1.
            relative_positions is assumed to have an arbitrary shape.
        sigma (float): the value of sigma. Should be smaller than the sigma threshold.
        kmax (int): largest positive integer in the sum. The sum is from -kmax to +kmax.

    Returns:
        list_sigma_normalized_score (torch.Tensor): the sigma_normalized_scores, in the same shape as
            relative_positions.

    """
    assert kmax >= 0, "kmax must be a non negative integer"
    assert 0 <= sigma <= SIGMA_THRESHOLD, "sigma should be between 0 and the 'small sigma' threshold"
    assert torch.logical_and(relative_positions >= 0, relative_positions < 1).all(), \
        "the relative positions should all be in [0, 1)"

    shape = relative_positions.shape

    column_u = relative_positions.flatten()[:, None]
    list_k = torch.arange(-kmax, kmax + 1)

    # Broadcast to shape [number of u values, number of k values]
    exponential = torch.exp(- 0.5 * (list_k ** 2 + 2. * column_u * list_k) / sigma**2)

    numerator = ((torch.ones_like(column_u) * list_k) * exponential).sum(dim=1)
    denominator = exponential.sum(dim=1)

    s1 = -relative_positions - (numerator / denominator).reshape(shape)
    return s1

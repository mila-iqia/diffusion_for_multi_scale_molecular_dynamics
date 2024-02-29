import numpy as np
import pytest
import torch

from crystal_diffusion.score.wrapped_gaussian_score import (
    SIGMA_THRESHOLD, get_sigma_normalized_score_small_sigma)


@pytest.fixture
def relative_positions(shape):
    torch.manual_seed(1234)
    return torch.rand(shape).to(torch.double)


def get_expected_sigma_normalized_score_small_sigma_brute_force(u: float, sigma: float, kmax: float):
    """A brute force implementation of S1 to check that main code is correct."""
    z1 = 0.
    sigma2_derivative_z1 = 0.
    for k in np.arange(-kmax, kmax + 1):
        upk = (u + k)
        exp = np.exp(-0.5 * upk ** 2 / sigma ** 2)

        z1 += exp
        sigma2_derivative_z1 += - upk * exp

    return sigma2_derivative_z1 / z1


@pytest.fixture
def expected_sigma_normalized_score_small_sigma(relative_positions, sigma, kmax):
    shape = relative_positions.shape

    list_s1 = []
    for u in relative_positions.numpy().flatten():
        s1 = get_expected_sigma_normalized_score_small_sigma_brute_force(u, sigma, kmax)
        list_s1.append(s1)

    return torch.tensor(list_s1).reshape(shape)


@pytest.mark.parametrize("sigma", [0.1 * SIGMA_THRESHOLD, 0.5 * SIGMA_THRESHOLD, 1 * SIGMA_THRESHOLD])
@pytest.mark.parametrize("kmax", [1, 10, 100])
@pytest.mark.parametrize("shape", [(10,), (3, 4, 5), (10, 5)])
def test_get_sigma_normalized_score_small_sigma(relative_positions, sigma, kmax,
                                                expected_sigma_normalized_score_small_sigma):
    sigma_normalized_score_small_sigma = get_sigma_normalized_score_small_sigma(relative_positions, sigma, kmax)

    torch.testing.assert_allclose(sigma_normalized_score_small_sigma, expected_sigma_normalized_score_small_sigma)

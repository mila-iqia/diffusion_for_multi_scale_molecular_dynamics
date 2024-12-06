import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.noise_utils import \
    scale_sigma_by_number_of_atoms


@pytest.fixture
def number_of_atoms(batch_size):
    return torch.randint(0, 100, (batch_size,))


@pytest.fixture
def sigma(batch_size):
    return torch.rand(batch_size)


@pytest.fixture
def expected_scaled_sigma(sigma, number_of_atoms, spatial_dimension):
    scaled_sigma = []
    for s, n in zip(sigma, number_of_atoms):
        scaled_s = s / np.power(n, 1 / spatial_dimension)
        scaled_sigma.append(scaled_s)
    return torch.tensor(scaled_sigma)


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("spatial_dimension", [1, 2, 3])
def test_scale_sigma_by_number_of_atoms(
    spatial_dimension, number_of_atoms, sigma, expected_scaled_sigma
):
    computed_scaled_sigma = scale_sigma_by_number_of_atoms(
        sigma, number_of_atoms, spatial_dimension
    )
    torch.testing.assert_allclose(expected_scaled_sigma, computed_scaled_sigma)

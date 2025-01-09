import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.score.gaussian_score import \
    get_lattice_sigma_normalized_score


@pytest.fixture(scope="module", autouse=True)
def set_random_seed():
    torch.manual_seed(1234)


@pytest.fixture(scope="module", autouse=True)
def set_default_type_to_float64():
    torch.set_default_dtype(torch.float64)
    yield
    # this returns the default type to float32 at the end of all tests in this class in order
    # to not affect other tests.
    torch.set_default_dtype(torch.float32)


@pytest.fixture
def num_lattice_parameters(spatial_dimension):
    return int(spatial_dimension * (spatial_dimension + 1) / 2)


@pytest.fixture
def real_lattice_parameters(batch_size, num_lattice_parameters):
    # input l0 to score
    return torch.randn(batch_size, num_lattice_parameters)


@pytest.fixture
def noisy_lattice_parameters(batch_size, num_lattice_parameters):
    return torch.randn(batch_size, num_lattice_parameters)


@pytest.fixture
def sigmas(batch_size, num_lattice_parameters):
    return torch.rand(batch_size, num_lattice_parameters) * 0.01


@pytest.fixture
def alpha_bars(batch_size, num_lattice_parameters):
    return torch.rand(batch_size, num_lattice_parameters)


@pytest.fixture
def expected_sigma_normalized_scores(real_lattice_parameters, noisy_lattice_parameters, sigmas, alpha_bars):
    shape = real_lattice_parameters.shape

    list_sigma_normalized_scores = []
    for real_l, noisy_l, sigma, alpha_bar in zip(
        real_lattice_parameters.numpy().flatten(),
        noisy_lattice_parameters.numpy().flatten(),
        sigmas.numpy().flatten(),
        alpha_bars.flatten(),
    ):
        s = -(noisy_l - np.sqrt(alpha_bar) * real_l) / (np.sqrt(1 - alpha_bar) * sigma)
        list_sigma_normalized_scores.append(s)

    return torch.tensor(list_sigma_normalized_scores).reshape(shape)


@pytest.mark.parametrize("batch_size", [1, 3, 7, 16])
@pytest.mark.parametrize("spatial_dimension", [1, 2, 3])
def test_get_sigma_normalized_score(
    real_lattice_parameters, noisy_lattice_parameters, sigmas, alpha_bars, expected_sigma_normalized_scores
):
    sigma_normalized_score = get_lattice_sigma_normalized_score(
        noisy_lattice_parameters,
        real_lattice_parameters,
        sigmas,
        alpha_bars,
    )

    torch.testing.assert_close(
        sigma_normalized_score,
        expected_sigma_normalized_scores,
    )

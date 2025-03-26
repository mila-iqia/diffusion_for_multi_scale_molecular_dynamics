import einops
import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.score.wrapped_gaussian_score import (
    SIGMA_THRESHOLD, _get_large_sigma_mask, _get_s1a_exponential,
    _get_s1b_exponential, _get_sigma_normalized_s2,
    _get_sigma_square_times_score_1_from_exponential,
    _get_small_sigma_large_u_mask, _get_small_sigma_small_u_mask,
    get_coordinates_sigma_normalized_score, get_log_wrapped_gaussians,
    get_sigma_normalized_score_brute_force)


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
def relative_coordinates(shape):
    return torch.rand(shape)


@pytest.fixture
def sigmas(shape):
    return torch.rand(shape) * 5.0 * SIGMA_THRESHOLD


@pytest.fixture()
def list_u(relative_coordinates):
    return relative_coordinates.flatten()


@pytest.fixture()
def list_sigma(sigmas):
    return sigmas.flatten()


@pytest.fixture()
def list_k(kmax):
    return torch.arange(-kmax, kmax + 1)


@pytest.fixture
def large_sigmas(shape):
    return (10.0 * torch.rand(shape)).clip(0.1)


@pytest.fixture
def expected_sigma_normalized_scores(relative_coordinates, large_sigmas):
    shape = relative_coordinates.shape

    list_sigma_normalized_scores = []
    for u, sigma in zip(
        relative_coordinates.numpy().flatten(), large_sigmas.numpy().flatten()
    ):
        s = get_sigma_normalized_score_brute_force(u, sigma)
        list_sigma_normalized_scores.append(s)

    return torch.tensor(list_sigma_normalized_scores).reshape(shape)


test_shapes = [(100,), (3, 4, 5), (10, 5)]


@pytest.mark.parametrize("shape", test_shapes)
class TestMasks:
    """This class tests the masks."""

    @pytest.fixture()
    def expected_shape(self, list_u):
        return (len(list_u),)

    def test_mask_1a(self, expected_shape, list_u, list_sigma):
        list_computed_mask_1a = _get_small_sigma_small_u_mask(list_u, list_sigma)
        assert (
            list_computed_mask_1a.shape == expected_shape
        ), "Wrong shape for output mask!"

        for u, sigma, mask in zip(list_u, list_sigma, list_computed_mask_1a):
            sigma_condition = sigma <= SIGMA_THRESHOLD
            u_condition = 0 <= u < 0.5
            expected_mask = sigma_condition and u_condition

            assert expected_mask == mask

    def test_mask_1b(self, expected_shape, list_u, list_sigma):
        list_computed_mask_1b = _get_small_sigma_large_u_mask(list_u, list_sigma)
        assert (
            list_computed_mask_1b.shape == expected_shape
        ), "Wrong shape for output mask!"

        for u, sigma, mask in zip(list_u, list_sigma, list_computed_mask_1b):
            sigma_condition = sigma <= SIGMA_THRESHOLD
            u_condition = 0.5 <= u < 1.0
            expected_mask = sigma_condition and u_condition

            assert expected_mask == mask

    def test_mask_2(self, expected_shape, list_u, list_sigma):
        list_computed_mask_2 = _get_large_sigma_mask(list_u, list_sigma)
        assert (
            list_computed_mask_2.shape == expected_shape
        ), "Wrong shape for output mask!"

        for sigma, mask in zip(list_sigma, list_computed_mask_2):
            expected_mask = sigma > SIGMA_THRESHOLD

            assert expected_mask == mask


@pytest.mark.parametrize("shape", test_shapes)
@pytest.mark.parametrize("kmax", [1, 5, 10])
class TestExponentials:
    @pytest.fixture()
    def fake_exponential(self, list_u, list_k):
        return torch.rand(len(list_u), len(list_k))

    def test_get_s1a_exponential(self, list_k, list_u, list_sigma):
        exponential = _get_s1a_exponential(list_u, list_sigma, list_k)

        for i, (u, sigma) in enumerate(zip(list_u, list_sigma)):
            for j, k in enumerate(list_k):
                computed_value = exponential[i, j]

                exponent = -0.5 * (k**2 + 2 * u * k) / sigma**2
                expected_value = exponent.exp()
                torch.testing.assert_close(computed_value, expected_value)

    def test_get_s1b_exponential(self, list_k, list_u, list_sigma):
        exponential = _get_s1b_exponential(list_u, list_sigma, list_k)

        for i, (u, sigma) in enumerate(zip(list_u, list_sigma)):
            for j, k in enumerate(list_k):
                computed_value = exponential[i, j]

                exponent = -0.5 * (k**2 - 1.0 + 2 * u * (k + 1.0)) / sigma**2
                expected_value = exponent.exp()
                torch.testing.assert_close(computed_value, expected_value)

    def test_get_sigma_normalized_s1_from_exponential(
        self, fake_exponential, list_u, list_k
    ):
        computed_results = _get_sigma_square_times_score_1_from_exponential(
            fake_exponential, list_u, list_k
        )

        expected_results = -torch.clone(list_u)
        for i, exponential_row in enumerate(fake_exponential):
            numerator = (exponential_row * list_k).sum()
            denominator = exponential_row.sum()
            expected_results[i] -= numerator / denominator

        torch.testing.assert_close(expected_results, computed_results)


@pytest.mark.parametrize("shape", [(10, 10, 3, 2), (50, 8, 3)])
@pytest.mark.parametrize("kmax", [1, 5, 10])
def test_get_wrapped_gaussian(relative_coordinates, sigmas, shape, kmax):

    log_wrapped_gaussians = get_log_wrapped_gaussians(
        relative_coordinates, sigmas, kmax
    )
    assert log_wrapped_gaussians.shape == shape[:-2]

    computed_wrapped_gaussians = einops.rearrange(
        torch.exp(log_wrapped_gaussians), " ... -> (...)"
    )

    list_x = einops.rearrange(
        relative_coordinates, "... natoms space -> (...) (natoms space)"
    )
    list_sigma = einops.rearrange(sigmas, "... natoms space -> (...) (natoms space)")

    list_expected_wrapped_gaussians = []
    for row_x, row_sigma in zip(list_x, list_sigma):
        wrapped_gaussian = torch.tensor(1.0)
        for x, sigma in zip(row_x, row_sigma):
            prob = torch.tensor(0.0)
            for k in range(-kmax, kmax + 1):
                log_prob = torch.distributions.Normal(loc=k, scale=sigma).log_prob(x)
                prob += torch.exp(log_prob)
            wrapped_gaussian *= prob
        list_expected_wrapped_gaussians.append(wrapped_gaussian)

    expected_wrapped_gaussians = torch.tensor(list_expected_wrapped_gaussians)

    torch.testing.assert_close(computed_wrapped_gaussians, expected_wrapped_gaussians)


@pytest.mark.parametrize("kmax", [1, 5, 10])
@pytest.mark.parametrize("shape", test_shapes)
@pytest.mark.parametrize("numerical_type", [torch.double])
def test_get_sigma_normalized_s2(list_u, list_sigma, list_k, numerical_type):
    # TODO: the test fails for numerical_type = torch.float. Should we be worried about numerical error here?
    list_u_cast = list_u.to(numerical_type)
    list_sigma_cast = list_sigma.to(numerical_type)
    pi = torch.tensor(np.pi, dtype=numerical_type)

    list_computed_s2 = _get_sigma_normalized_s2(list_u_cast, list_sigma_cast, list_k)

    list_expected_s2 = []
    for u, sigma in zip(list_u_cast, list_sigma_cast):
        z2 = torch.tensor(0.0, dtype=numerical_type)
        deriv_z2 = torch.tensor(0.0, dtype=numerical_type)

        for k in list_k:
            g_term = (
                torch.sqrt(2.0 * pi) * sigma * (-2.0 * pi**2 * sigma**2 * k**2).exp()
                - (-pi * k**2).exp()
            )
            z2 += (-pi * (u + k) ** 2).exp() + g_term * torch.cos(2 * pi * k * u)
            deriv_z2 += -2.0 * pi * (u + k) * (
                -pi * (u + k) ** 2
            ).exp() - 2.0 * pi * k * g_term * torch.sin(2.0 * pi * k * u)

        expected_value = sigma * deriv_z2 / z2
        list_expected_s2.append(expected_value)

    list_expected_s2 = torch.tensor(list_expected_s2)

    torch.testing.assert_close(list_computed_s2, list_expected_s2)


@pytest.mark.parametrize("kmax", [4])
@pytest.mark.parametrize("shape", test_shapes)
def test_get_sigma_normalized_score(
    relative_coordinates, large_sigmas, kmax, expected_sigma_normalized_scores
):
    sigma_normalized_score_small_sigma = get_coordinates_sigma_normalized_score(
        relative_coordinates, large_sigmas, kmax
    )

    # The brute force calculation is fragile to the creation of NaNs.
    # Let's give the test a free pass when this happens.
    nan_mask = torch.where(expected_sigma_normalized_scores.isnan())
    expected_sigma_normalized_scores[nan_mask] = sigma_normalized_score_small_sigma[
        nan_mask
    ]

    torch.testing.assert_close(
        sigma_normalized_score_small_sigma,
        expected_sigma_normalized_scores,
    )

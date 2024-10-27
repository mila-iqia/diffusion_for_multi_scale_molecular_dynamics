import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import \
    ExplodingVariance
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters


class TestExplodingVariance:

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(23423)

    @pytest.fixture()
    def sigma_min(self):
        return 0.001

    @pytest.fixture()
    def sigma_max(self):
        return 0.5

    @pytest.fixture()
    def noise_parameters(self, sigma_min, sigma_max):
        return NoiseParameters(
            total_time_steps=10, sigma_min=sigma_min, sigma_max=sigma_max
        )

    @pytest.fixture()
    def times(self):
        return torch.rand(100)

    @pytest.fixture()
    def exploding_variance(self, noise_parameters):
        return ExplodingVariance(noise_parameters)

    @pytest.fixture()
    def expected_sigmas(self, noise_parameters, times):
        expected_sigmas = []
        for t in times:
            sigma = (
                noise_parameters.sigma_min ** (1.0 - t) * noise_parameters.sigma_max**t
            )
            expected_sigmas.append(sigma)

        return torch.tensor(expected_sigmas)

    def test_get_sigma(self, exploding_variance, times, expected_sigmas):
        computed_sigmas = exploding_variance.get_sigma(times)
        torch.testing.assert_close(computed_sigmas, expected_sigmas)

    def test_get_sigma_time_derivative(self, exploding_variance, times):
        computed_sigmas_dot = exploding_variance.get_sigma_time_derivative(times)

        t = torch.tensor(times, requires_grad=True)
        sigma = exploding_variance.get_sigma(t)

        gradients = torch.ones_like(sigma)
        sigma.backward(gradients)

        expected_sigma_dot = t.grad
        torch.testing.assert_close(computed_sigmas_dot, expected_sigma_dot)

    def test_get_g_squared(self, exploding_variance, times):
        computed_g_squared = exploding_variance.get_g_squared(times)

        t = torch.tensor(times, requires_grad=True)
        sigma = exploding_variance.get_sigma(t)

        sigma_squared = sigma**2

        gradients = torch.ones_like(sigma_squared)
        sigma_squared.backward(gradients)
        expected_g_squared = t.grad

        torch.testing.assert_close(expected_g_squared, computed_g_squared)

import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.sigma_calculator import (
    ExponentialSigmaCalculator, LinearSigmaCalculator)


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
    def times(self):
        return torch.rand(100)

    @pytest.fixture(params=["exponential", "linear"])
    def schedule_type(self, request):
        return request.param

    @pytest.fixture()
    def expected_exponential(self, sigma_min, sigma_max, times):
        expected_sigmas = []
        expected_sigmas_time_derivatives = []
        ratio = np.log(sigma_max / sigma_min)
        for t in times:
            sigma = sigma_min ** (1.0 - t) * sigma_max**t
            expected_sigmas.append(sigma)

            d_sigma_dt = sigma * ratio
            expected_sigmas_time_derivatives.append(d_sigma_dt)

        return torch.tensor(expected_sigmas), torch.tensor(expected_sigmas_time_derivatives)

    @pytest.fixture()
    def expected_linear(self, sigma_min, sigma_max, times):
        expected_sigmas = []
        expected_sigmas_time_derivatives = []
        for t in times:
            sigma = sigma_min + (sigma_max - sigma_min) * t
            expected_sigmas.append(sigma)
            expected_sigmas_time_derivatives.append(sigma_max - sigma_min)

        return torch.tensor(expected_sigmas), torch.tensor(expected_sigmas_time_derivatives)

    @pytest.fixture()
    def expected_values(self, schedule_type, expected_exponential, expected_linear):
        if schedule_type == "exponential":
            return expected_exponential
        elif schedule_type == "linear":
            return expected_linear
        else:
            raise NotImplementedError

    @pytest.fixture()
    def sigma_calculator(self, schedule_type, sigma_min, sigma_max):
        if schedule_type == "exponential":
            return ExponentialSigmaCalculator(sigma_min, sigma_max)
        elif schedule_type == "linear":
            return LinearSigmaCalculator(sigma_min, sigma_max)
        else:
            raise NotImplementedError("Unknown schedule_type")

    def test_sigma_calculator(self, sigma_calculator, times, expected_values):
        expected_sigmas, expected_sigma_time_derivatives = expected_values

        torch.testing.assert_close(expected_sigmas, sigma_calculator.get_sigma(times))
        torch.testing.assert_close(expected_sigma_time_derivatives, sigma_calculator.get_sigma_time_derivative(times))

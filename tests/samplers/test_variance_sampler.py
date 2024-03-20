import pytest
import torch

from crystal_diffusion.samplers.variance_sampler import (
    ExplodingVarianceSampler, NoiseParameters)


@pytest.mark.parametrize("total_time_steps", [3, 10, 17])
class TestExplodingVarianceSampler:
    @pytest.fixture()
    def noise_parameters(self, total_time_steps):
        return NoiseParameters(total_time_steps=total_time_steps)

    @pytest.fixture()
    def variance_sampler(self, noise_parameters):
        return ExplodingVarianceSampler(noise_parameters=noise_parameters)

    @pytest.fixture()
    def expected_times(self, total_time_steps):
        times = torch.linspace(0.0, 1.0, total_time_steps)
        return times

    @pytest.fixture()
    def expected_sigmas(self, expected_times, noise_parameters):
        smin = noise_parameters.sigma_min
        smax = noise_parameters.sigma_max

        sigmas = smin ** (1.0 - expected_times) * smax**expected_times
        return sigmas

    @pytest.fixture()
    def indices(self, time_sampler, shape):
        return time_sampler.get_random_time_step_indices(shape)

    def test_time_array(self, variance_sampler, expected_times):
        torch.testing.assert_allclose(variance_sampler._time_array, expected_times)

    def test_sigma_and_sigma_squared_arrays(self, variance_sampler, expected_sigmas):
        torch.testing.assert_allclose(variance_sampler._sigma_array, expected_sigmas)
        torch.testing.assert_allclose(variance_sampler._sigma_squared_array, expected_sigmas**2)

    def test_g_and_g_square_array(self, variance_sampler, expected_sigmas):
        expected_sigmas_square = expected_sigmas**2

        expected_g_squared_array = [float("nan")]
        for sigma2_t, sigma2_tm1 in zip(
            expected_sigmas_square[1:], expected_sigmas_square[:-1]
        ):
            g2 = sigma2_t - sigma2_tm1
            expected_g_squared_array.append(g2)

        expected_g_squared_array = torch.tensor(expected_g_squared_array)
        expected_g_array = torch.sqrt(expected_g_squared_array)

        assert torch.isnan(variance_sampler._g_array[0])
        assert torch.isnan(variance_sampler._g_squared_array[0])
        torch.testing.assert_allclose(variance_sampler._g_array[1:], expected_g_array[1:])
        torch.testing.assert_allclose(variance_sampler._g_squared_array[1:], expected_g_squared_array[1:])

    def test_get_random_time_step_indices(self, variance_sampler, total_time_steps):
        # Check that we never sample zero.
        random_indices = variance_sampler._get_random_time_step_indices(shape=(1000,))
        assert torch.all(random_indices > 0)
        assert torch.all(random_indices < total_time_steps)

    @pytest.mark.parametrize("batch_size", [1, 10, 100])
    def test_get_random_noise_parameter_sample(
        self, mocker, variance_sampler, batch_size
    ):
        random_indices = variance_sampler._get_random_time_step_indices(shape=(1000,))
        mocker.patch.object(
            variance_sampler,
            "_get_random_time_step_indices",
            return_value=random_indices,
        )

        noise_sample = variance_sampler.get_random_noise_sample(batch_size)

        expected_times = variance_sampler._time_array.take(random_indices)
        expected_sigmas = variance_sampler._sigma_array.take(random_indices)
        expected_sigmas_squared = variance_sampler._sigma_squared_array.take(
            random_indices
        )
        expected_gs = variance_sampler._g_array.take(random_indices)
        expected_gs_squared = variance_sampler._g_squared_array.take(random_indices)

        torch.testing.assert_allclose(noise_sample.time, expected_times)
        torch.testing.assert_allclose(noise_sample.sigma, expected_sigmas)
        torch.testing.assert_allclose(noise_sample.sigma_squared, expected_sigmas_squared)
        torch.testing.assert_allclose(noise_sample.g, expected_gs)
        torch.testing.assert_allclose(noise_sample.g_squared, expected_gs_squared)

    def test_get_all_noise(self, variance_sampler):
        noise = variance_sampler.get_all_noise()
        torch.testing.assert_allclose(noise.time, variance_sampler._time_array)
        torch.testing.assert_allclose(noise.sigma, variance_sampler._sigma_array)
        torch.testing.assert_allclose(noise.sigma_squared, variance_sampler._sigma_squared_array)
        assert torch.isnan(noise.g[0])
        assert torch.isnan(noise.g_squared[0])
        torch.testing.assert_allclose(noise.g[1:], variance_sampler._g_array[1:])
        torch.testing.assert_allclose(noise.g_squared[1:], variance_sampler._g_squared_array[1:])

import pytest
import torch

from src.diffusion_for_multi_scale_molecular_dynamics.samplers.variance_sampler import (
    ExplodingVarianceSampler, NoiseParameters)


@pytest.mark.parametrize("total_time_steps", [3, 10, 17])
@pytest.mark.parametrize("time_delta", [1e-5, 0.1])
@pytest.mark.parametrize("sigma_min", [0.005, 0.1])
@pytest.mark.parametrize("corrector_step_epsilon", [2e-5, 0.1])
class TestExplodingVarianceSampler:
    @pytest.fixture()
    def noise_parameters(
        self, total_time_steps, time_delta, sigma_min, corrector_step_epsilon
    ):
        return NoiseParameters(
            total_time_steps=total_time_steps,
            time_delta=time_delta,
            sigma_min=sigma_min,
            corrector_step_epsilon=corrector_step_epsilon,
        )

    @pytest.fixture()
    def variance_sampler(self, noise_parameters):
        return ExplodingVarianceSampler(noise_parameters=noise_parameters)

    @pytest.fixture()
    def expected_times(self, total_time_steps, time_delta):
        times = []
        for i in range(total_time_steps):
            t = i / (total_time_steps - 1) * (1.0 - time_delta) + time_delta
            times.append(t)
        times = torch.tensor(times)
        return times

    @pytest.fixture()
    def expected_sigmas(self, expected_times, noise_parameters):
        smin = noise_parameters.sigma_min
        smax = noise_parameters.sigma_max

        sigmas = smin ** (1.0 - expected_times) * smax**expected_times
        return sigmas

    @pytest.fixture()
    def expected_epsilons(self, expected_sigmas, noise_parameters):
        smin = noise_parameters.sigma_min
        eps = noise_parameters.corrector_step_epsilon

        s1 = expected_sigmas[0]

        epsilons = [0.5 * eps * smin**2 / s1**2]
        for i in range(len(expected_sigmas) - 1):
            si = expected_sigmas[i]
            epsilons.append(0.5 * eps * si**2 / s1**2)

        return torch.tensor(epsilons)

    @pytest.fixture()
    def indices(self, time_sampler, shape):
        return time_sampler.get_random_time_step_indices(shape)

    def test_time_array(self, variance_sampler, expected_times):
        torch.testing.assert_close(variance_sampler._time_array, expected_times)

    def test_sigma_and_sigma_squared_arrays(self, variance_sampler, expected_sigmas):
        torch.testing.assert_close(variance_sampler._sigma_array, expected_sigmas)
        torch.testing.assert_close(
            variance_sampler._sigma_squared_array, expected_sigmas**2
        )

    def test_g_and_g_square_array(self, variance_sampler, expected_sigmas, sigma_min):
        expected_sigmas_square = expected_sigmas**2

        sigma1 = torch.sqrt(expected_sigmas_square[0])

        expected_g_squared_array = [sigma1**2 - sigma_min**2]
        for sigma2_t, sigma2_tm1 in zip(
            expected_sigmas_square[1:], expected_sigmas_square[:-1]
        ):
            g2 = sigma2_t - sigma2_tm1
            expected_g_squared_array.append(g2)

        expected_g_squared_array = torch.tensor(expected_g_squared_array)
        expected_g_array = torch.sqrt(expected_g_squared_array)

        torch.testing.assert_close(variance_sampler._g_array, expected_g_array)
        torch.testing.assert_close(
            variance_sampler._g_squared_array, expected_g_squared_array
        )

    def test_epsilon_arrays(self, variance_sampler, expected_epsilons):
        torch.testing.assert_close(variance_sampler._epsilon_array, expected_epsilons)
        torch.testing.assert_close(
            variance_sampler._sqrt_two_epsilon_array,
            torch.sqrt(2.0 * expected_epsilons),
        )

    def test_get_random_time_step_indices(self, variance_sampler, total_time_steps):
        random_indices = variance_sampler._get_random_time_step_indices(shape=(1000,))
        assert torch.all(random_indices >= 0)
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

        torch.testing.assert_close(noise_sample.time, expected_times)
        torch.testing.assert_close(noise_sample.sigma, expected_sigmas)
        torch.testing.assert_close(noise_sample.sigma_squared, expected_sigmas_squared)
        torch.testing.assert_close(noise_sample.g, expected_gs)
        torch.testing.assert_close(noise_sample.g_squared, expected_gs_squared)

    def test_get_all_sampling_parameters(self, variance_sampler):
        noise, langevin_dynamics = variance_sampler.get_all_sampling_parameters()
        torch.testing.assert_close(noise.time, variance_sampler._time_array)
        torch.testing.assert_close(noise.sigma, variance_sampler._sigma_array)
        torch.testing.assert_close(
            noise.sigma_squared, variance_sampler._sigma_squared_array
        )
        torch.testing.assert_close(noise.g, variance_sampler._g_array)
        torch.testing.assert_close(noise.g_squared, variance_sampler._g_squared_array)

        torch.testing.assert_close(
            langevin_dynamics.epsilon, variance_sampler._epsilon_array
        )
        torch.testing.assert_close(
            langevin_dynamics.sqrt_2_epsilon, variance_sampler._sqrt_two_epsilon_array
        )

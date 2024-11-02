import einops
import pytest
import torch

from src.diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import (
    NoiseParameters,
)
from src.diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.variance_sampler import (
    NoiseScheduler,
)


@pytest.mark.parametrize("total_time_steps", [3, 10, 17])
@pytest.mark.parametrize("time_delta", [1e-5, 0.1])
@pytest.mark.parametrize("sigma_min", [0.005, 0.1])
@pytest.mark.parametrize("corrector_step_epsilon", [2e-5, 0.1])
@pytest.mark.parametrize("num_classes", [4])
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
    def variance_sampler(self, noise_parameters, num_classes):
        return NoiseScheduler(
            noise_parameters=noise_parameters, num_classes=num_classes
        )

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
    def expected_betas(self, expected_times, noise_parameters):
        betas = []
        for i in range(noise_parameters.total_time_steps):
            betas.append(1.0 / (noise_parameters.total_time_steps - i))
        return torch.tensor(betas)

    @pytest.fixture()
    def expected_alphas(self, expected_betas):
        alphas = [1 - expected_betas[0]]
        for beta in expected_betas[1:]:
            alphas.append(alphas[-1] * (1 - beta.item()))
        return torch.tensor(alphas)

    @pytest.fixture()
    def expected_q_matrix(self, expected_betas, num_classes):
        expected_qs = []
        for beta in expected_betas:
            q = torch.zeros(1, num_classes, num_classes)
            for i in range(num_classes):
                q[0, i, i] = 1 - beta.item()
            q[0, :-1, -1] = beta.item()
            q[0, -1, -1] = 1
            expected_qs.append(q)
        return torch.concatenate(expected_qs, dim=0)

    @pytest.fixture()
    def expected_q_bar_matrix(self, expected_q_matrix):
        expected_qbars = [expected_q_matrix[0]]
        for qmat in expected_q_matrix[1:]:
            expected_qbars.append(
                einops.einsum(expected_qbars[-1], qmat, "i j, j k -> i k")
            )
        return torch.stack(expected_qbars, dim=0)

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

    def test_create_beta_array(self, variance_sampler, expected_betas):
        assert torch.allclose(variance_sampler._beta_array, expected_betas)

    def test_create_alpha_bar_array(self, variance_sampler, expected_alphas):
        assert torch.allclose(variance_sampler._alpha_bar_array, expected_alphas)

    def test_create_q_matrix_array(self, variance_sampler, expected_q_matrix):
        assert torch.allclose(variance_sampler._q_matrix_array, expected_q_matrix)

    def test_create_q_bar_matrix_array(self, variance_sampler, expected_q_bar_matrix):
        assert torch.allclose(
            variance_sampler._q_bar_matrix_array, expected_q_bar_matrix
        )

    @pytest.mark.parametrize("batch_size", [1, 10, 100])
    def test_get_random_noise_parameter_sample(
        self, mocker, variance_sampler, batch_size, num_classes
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
        expected_betas = variance_sampler._beta_array.take(random_indices)
        expected_alpha_bars = variance_sampler._alpha_bar_array.take(random_indices)
        expected_q_matrices = variance_sampler._q_matrix_array.index_select(
            dim=0, index=random_indices
        )
        expected_q_bar_matrices = variance_sampler._q_bar_matrix_array.index_select(
            dim=0, index=random_indices
        )
        expected_q_bar_tm1_matrices = torch.where(
            random_indices.view(-1, 1, 1) == 0,
            torch.eye(num_classes).unsqueeze(0),  # replace t=0 with identity matrix
            variance_sampler._q_bar_matrix_array.index_select(
                dim=0, index=(random_indices - 1).clip(min=0)
            ),
        )

        torch.testing.assert_close(noise_sample.time, expected_times)
        torch.testing.assert_close(noise_sample.sigma, expected_sigmas)
        torch.testing.assert_close(noise_sample.sigma_squared, expected_sigmas_squared)
        torch.testing.assert_close(noise_sample.g, expected_gs)
        torch.testing.assert_close(noise_sample.g_squared, expected_gs_squared)
        torch.testing.assert_close(noise_sample.beta, expected_betas)
        torch.testing.assert_close(noise_sample.alpha_bar, expected_alpha_bars)
        torch.testing.assert_close(noise_sample.q_matrix, expected_q_matrices)
        torch.testing.assert_close(noise_sample.q_bar_matrix, expected_q_bar_matrices)
        torch.testing.assert_close(
            noise_sample.q_bar_tm1_matrix, expected_q_bar_tm1_matrices
        )

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

        torch.testing.assert_close(noise.beta, variance_sampler._beta_array)
        torch.testing.assert_close(noise.alpha_bar, variance_sampler._alpha_bar_array)
        torch.testing.assert_close(noise.q_matrix, variance_sampler._q_matrix_array)
        torch.testing.assert_close(
            noise.q_bar_matrix, variance_sampler._q_bar_matrix_array
        )
        torch.testing.assert_close(
            noise.q_bar_tm1_matrix[1:], variance_sampler._q_bar_matrix_array[:-1]
        )

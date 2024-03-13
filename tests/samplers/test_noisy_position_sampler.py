import numpy as np
import pytest
import torch

from crystal_diffusion.samplers.noisy_position_sampler import \
    NoisyPositionSampler


@pytest.mark.parametrize("shape", [(10, 1), (4, 5, 3), (2, 2, 2, 2)])
class TestNoisyPositionSampler:
    @pytest.fixture()
    def real_relative_positions(self, shape):
        torch.manual_seed(23423)
        return torch.rand(shape)

    @pytest.fixture()
    def computed_noisy_relative_positions(self, real_relative_positions, sigma):
        return NoisyPositionSampler.get_noisy_position_sample(real_relative_positions, sigma)

    @pytest.fixture()
    def fake_gaussian_sample(self, real_relative_positions):
        # Note: this is NOT a Gaussian distribution. That's ok, it's fake data for testing!
        return torch.rand(real_relative_positions.shape)

    @pytest.mark.parametrize("sigma", [0.001, 0.01, 0.1, 1., 10.])
    def test_shape(self, computed_noisy_relative_positions, shape):
        assert computed_noisy_relative_positions.shape == shape

    @pytest.mark.parametrize("sigma", [0.001, 0.01, 0.1, 1., 10.])
    def test_range(self, computed_noisy_relative_positions):
        assert torch.all(computed_noisy_relative_positions >= 0.)
        assert torch.all(computed_noisy_relative_positions < 1.)

    @pytest.mark.parametrize("sigma", [0.0, 1e-8])
    def test_small_sigma_limit(self, computed_noisy_relative_positions, real_relative_positions):
        assert torch.all(torch.isclose(real_relative_positions, computed_noisy_relative_positions))

    @pytest.mark.parametrize("sigma", [0.001, 0.01, 0.1, 1., 10.])
    def test_get_noisy_position_sample(self, mocker, real_relative_positions, sigma, fake_gaussian_sample):
        mocker.patch.object(NoisyPositionSampler, "_get_gaussian_noise", return_value=fake_gaussian_sample)

        flat_computed_samples = NoisyPositionSampler.get_noisy_position_sample(real_relative_positions, sigma).flatten()

        flat_expected_samples = []
        for x0, epsilon in zip(real_relative_positions.flatten(), fake_gaussian_sample.flatten()):
            x = np.mod(x0 + sigma * epsilon, 1).float()
            flat_expected_samples.append(x)

        flat_expected_samples = torch.tensor(flat_expected_samples)

        assert torch.all(torch.isclose(flat_computed_samples, flat_expected_samples))

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
    def sigmas(self, shape):
        return torch.rand(shape)

    @pytest.fixture()
    def computed_noisy_relative_positions(self, real_relative_positions, sigmas):
        return NoisyPositionSampler.get_noisy_position_sample(
            real_relative_positions, sigmas
        )

    @pytest.fixture()
    def fake_gaussian_sample(self, shape):
        # Note: this is NOT a Gaussian distribution. That's ok, it's fake data for testing!
        return torch.rand(shape)

    def test_shape(self, computed_noisy_relative_positions, shape):
        assert computed_noisy_relative_positions.shape == shape

    def test_range(self, computed_noisy_relative_positions):
        assert torch.all(computed_noisy_relative_positions >= 0.0)
        assert torch.all(computed_noisy_relative_positions < 1.0)

    def test_get_noisy_position_sample(
        self, mocker, real_relative_positions, sigmas, fake_gaussian_sample
    ):
        mocker.patch.object(
            NoisyPositionSampler,
            "_get_gaussian_noise",
            return_value=fake_gaussian_sample,
        )

        computed_samples = NoisyPositionSampler.get_noisy_position_sample(
            real_relative_positions, sigmas
        )

        flat_sigmas = sigmas.flatten()
        flat_positions = real_relative_positions.flatten()
        flat_computed_samples = computed_samples.flatten()
        flat_fake_gaussian_sample = fake_gaussian_sample.flatten()

        for sigma, x0, computed_sample, epsilon in zip(
            flat_sigmas,
            flat_positions,
            flat_computed_samples,
            flat_fake_gaussian_sample,
        ):
            expected_sample = np.mod(x0 + sigma * epsilon, 1).float()

            assert torch.all(torch.isclose(computed_sample, expected_sample))

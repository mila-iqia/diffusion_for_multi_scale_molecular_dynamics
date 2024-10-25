import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.noisy_configurations.noisy_relative_coordinates import \
    NoisyRelativeCoordinates


@pytest.mark.parametrize("shape", [(10, 1), (4, 5, 3), (2, 2, 2, 2)])
class TestNoisyRelativeCoordinatesSampler:

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(23423)

    @pytest.fixture()
    def real_relative_coordinates(self, shape):
        return torch.rand(shape)

    @pytest.fixture()
    def sigmas(self, shape):
        return torch.rand(shape)

    @pytest.fixture()
    def computed_noisy_relative_coordinates(self, real_relative_coordinates, sigmas):
        return NoisyRelativeCoordinates.get_noisy_relative_coordinates_sample(
            real_relative_coordinates, sigmas
        )

    @pytest.fixture()
    def fake_gaussian_sample(self, shape):
        # Note: this is NOT a Gaussian distribution. That's ok, it's fake data for testing!
        return torch.rand(shape)

    def test_shape(self, computed_noisy_relative_coordinates, shape):
        assert computed_noisy_relative_coordinates.shape == shape

    def test_range(self, computed_noisy_relative_coordinates):
        assert torch.all(computed_noisy_relative_coordinates >= 0.0)
        assert torch.all(computed_noisy_relative_coordinates < 1.0)

    def test_get_noisy_relative_coordinates_sample(
        self, mocker, real_relative_coordinates, sigmas, fake_gaussian_sample
    ):
        mocker.patch.object(
            NoisyRelativeCoordinates,
            "_get_gaussian_noise",
            return_value=fake_gaussian_sample,
        )

        computed_samples = (
            NoisyRelativeCoordinates.get_noisy_relative_coordinates_sample(
                real_relative_coordinates, sigmas
            )
        )

        flat_sigmas = sigmas.flatten()
        flat_relative_coordinates = real_relative_coordinates.flatten()
        flat_computed_samples = computed_samples.flatten()
        flat_fake_gaussian_sample = fake_gaussian_sample.flatten()

        for sigma, x0, computed_sample, epsilon in zip(
            flat_sigmas,
            flat_relative_coordinates,
            flat_computed_samples,
            flat_fake_gaussian_sample,
        ):
            expected_sample = np.mod(x0 + sigma * epsilon, 1).float()

            torch.testing.assert_close(computed_sample, expected_sample)

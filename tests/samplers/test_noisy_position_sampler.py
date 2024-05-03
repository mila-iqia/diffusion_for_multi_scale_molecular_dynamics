import numpy as np
import pytest
import torch

from crystal_diffusion.samplers.noisy_position_sampler import (
    NoisyPositionSampler, map_positions_to_unit_cell)


def test_remainder_failure():
    # This test demonstrates how torch.remainder does not do what we want.
    epsilon = -torch.tensor(1.e-8)
    position_not_in_unit_cell = torch.remainder(epsilon, 1.0)
    assert position_not_in_unit_cell == 1.0


@pytest.mark.parametrize("shape", [(10,), (10, 20), (3, 4, 5)])
def test_map_positions_to_unit_cell_hard(shape):
    positions = 1e-8 * (torch.rand((10,)) - 0.5)
    computed_positions = map_positions_to_unit_cell(positions)

    positive_positions_mask = positions >= 0.
    assert torch.all(positions[positive_positions_mask] == computed_positions[positive_positions_mask])
    torch.testing.assert_close(computed_positions[~positive_positions_mask],
                               torch.zeros_like(computed_positions[~positive_positions_mask]))


@pytest.mark.parametrize("shape", [(100, 8, 16)])
def test_map_positions_to_unit_cell_easy(shape):
    # Very unlikely to hit the edge cases.
    positions = 10. * (torch.rand((10,)) - 0.5)
    expected_values = torch.remainder(positions, 1.)
    computed_values = map_positions_to_unit_cell(positions)
    torch.testing.assert_close(computed_values, expected_values)


@pytest.mark.parametrize("shape", [(10, 1), (4, 5, 3), (2, 2, 2, 2)])
class TestNoisyPositionSampler:

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(23423)

    @pytest.fixture()
    def real_relative_positions(self, shape):
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

            torch.testing.assert_close(computed_sample, expected_sample)

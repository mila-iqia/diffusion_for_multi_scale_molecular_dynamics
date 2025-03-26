import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.noisers.lattice_noiser import (
    LatticeDataParameters, LatticeNoiser)


@pytest.mark.parametrize("spatial_dimension", [1, 2, 3])
class TestLatticeNoiser:

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(23423)

    @pytest.fixture()
    def num_lattice_parameters(self, spatial_dimension):
        return int(spatial_dimension * (spatial_dimension + 1) / 2)

    @pytest.fixture()
    def batch_size(self):
        return 16

    @pytest.fixture()
    def lattice_parameters(self, spatial_dimension):
        return LatticeDataParameters(
            spatial_dimension=spatial_dimension,
        )

    @pytest.fixture()
    def lattice_noiser(self, lattice_parameters):
        return LatticeNoiser(lattice_parameters)

    @pytest.fixture()
    def real_lattice_parameters(self, batch_size, num_lattice_parameters):
        return torch.rand(batch_size, num_lattice_parameters)

    @pytest.fixture()
    def sigmas(self, batch_size, num_lattice_parameters):
        return torch.rand(batch_size, num_lattice_parameters)

    @pytest.fixture()
    def computed_noisy_lattice_parameters(
        self,
        lattice_noiser,
        real_lattice_parameters,
        sigmas,
    ):
        return lattice_noiser.get_noisy_lattice_parameters(real_lattice_parameters, sigmas)

    @pytest.fixture()
    def fake_gaussian_sample(self, batch_size, num_lattice_parameters):
        # Note: this is NOT a Gaussian distribution. That's ok, it's fake data for testing!
        return torch.rand(batch_size, num_lattice_parameters)

    def test_shape(
        self, computed_noisy_lattice_parameters, batch_size, num_lattice_parameters
    ):
        assert computed_noisy_lattice_parameters.shape == (
            batch_size,
            num_lattice_parameters,
        )

    def test_get_noisy_lattice_parameters_sample(
        self,
        mocker,
        real_lattice_parameters,
        sigmas,
        fake_gaussian_sample,
        lattice_noiser,
        spatial_dimension,
    ):
        mocker.patch.object(
            lattice_noiser,
            "_get_gaussian_noise",
            return_value=fake_gaussian_sample,
        )

        computed_samples = lattice_noiser.get_noisy_lattice_parameters(
            real_lattice_parameters,
            sigmas,
        )

        flat_sigmas = sigmas.flatten()
        flat_lattice_parameters = real_lattice_parameters.flatten()
        flat_computed_samples = computed_samples.flatten()
        flat_fake_gaussian_sample = fake_gaussian_sample.flatten()

        for sigma, l0, computed_sample, epsilon in zip(
            flat_sigmas,
            flat_lattice_parameters,
            flat_computed_samples,
            flat_fake_gaussian_sample,
        ):
            sample_noisy_part = epsilon * sigma
            expected_sample = sample_noisy_part + l0

            torch.testing.assert_close(computed_sample, expected_sample)

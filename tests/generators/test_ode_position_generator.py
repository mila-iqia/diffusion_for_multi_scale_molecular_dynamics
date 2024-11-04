import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.ode_position_generator import (
    ExplodingVarianceODEPositionGenerator, ODESamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from src.diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.variance_sampler import \
    NoiseScheduler
from tests.generators.conftest import BaseTestGenerator


@pytest.mark.parametrize("total_time_steps", [2, 5, 10])
@pytest.mark.parametrize("sigma_min", [0.15])
@pytest.mark.parametrize("record_samples", [False, True])
@pytest.mark.parametrize("number_of_samples", [8])
class TestExplodingVarianceODEPositionGenerator(BaseTestGenerator):

    @pytest.fixture()
    def noise_parameters(self, total_time_steps, sigma_min):
        return NoiseParameters(
            total_time_steps=total_time_steps, time_delta=0.0, sigma_min=sigma_min
        )

    @pytest.fixture()
    def sampling_parameters(
        self,
        number_of_atoms,
        spatial_dimension,
        cell_dimensions,
        number_of_samples,
        record_samples,
        num_atom_types,
    ):
        sampling_parameters = ODESamplingParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            number_of_samples=number_of_samples,
            cell_dimensions=cell_dimensions,
            record_samples=record_samples,
            num_atom_types=num_atom_types,
        )
        return sampling_parameters

    @pytest.fixture()
    def ode_generator(
        self, noise_parameters, sampling_parameters, sigma_normalized_score_network
    ):
        generator = ExplodingVarianceODEPositionGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            sigma_normalized_score_network=sigma_normalized_score_network,
        )

        return generator

    def test_get_ode_prefactor(self, ode_generator, noise_parameters):
        times = NoiseScheduler._get_time_array(noise_parameters)
        sigmas = (
            noise_parameters.sigma_min ** (1.0 - times)
            * noise_parameters.sigma_max**times
        )

        sig_ratio = torch.tensor(
            noise_parameters.sigma_max / noise_parameters.sigma_min
        )
        expected_ode_prefactor = torch.log(sig_ratio) * sigmas
        computed_ode_prefactor = ode_generator._get_ode_prefactor(times)
        torch.testing.assert_close(expected_ode_prefactor, computed_ode_prefactor)

    def test_smoke_sample(
        self,
        ode_generator,
        device,
        number_of_samples,
        number_of_atoms,
        spatial_dimension,
        unit_cell_sample,
    ):
        # Just a smoke test that we can sample without crashing.
        relative_coordinates = ode_generator.sample(
            number_of_samples, device, unit_cell_sample
        )

        assert relative_coordinates.shape == (
            number_of_samples,
            number_of_atoms,
            spatial_dimension,
        )

        assert relative_coordinates.min() >= 0.0
        assert relative_coordinates.max() < 1.0

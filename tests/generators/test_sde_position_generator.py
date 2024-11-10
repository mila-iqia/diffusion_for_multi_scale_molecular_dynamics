import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.sde_position_generator import (
    SDE, ExplodingVarianceSDEPositionGenerator, SDESamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import \
    VarianceScheduler
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from tests.generators.conftest import BaseTestGenerator


@pytest.mark.parametrize("total_time_steps", [5, 10])
@pytest.mark.parametrize("sigma_min", [0.15])
@pytest.mark.parametrize("record_samples", [False, True])
class TestExplodingVarianceSDEPositionGenerator(BaseTestGenerator):
    @pytest.fixture()
    def initial_diffusion_time(self):
        return torch.tensor(0.0)

    @pytest.fixture()
    def final_diffusion_time(self):
        return torch.tensor(1.0)

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
        sampling_parameters = SDESamplingParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            number_of_samples=number_of_samples,
            cell_dimensions=cell_dimensions,
            record_samples=record_samples,
            num_atom_types=num_atom_types,
        )
        return sampling_parameters

    @pytest.fixture()
    def atom_types(self, number_of_samples, number_of_atoms):
        return torch.zeros(number_of_samples, number_of_atoms).long()

    @pytest.fixture()
    def sde(
        self,
        noise_parameters,
        sampling_parameters,
        axl_network,
        atom_types,
        unit_cell_sample,
        initial_diffusion_time,
        final_diffusion_time,
    ):
        sde = SDE(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=axl_network,
            atom_types=atom_types,
            unit_cells=unit_cell_sample,
            initial_diffusion_time=initial_diffusion_time,
            final_diffusion_time=final_diffusion_time,
        )
        return sde

    def test_sde_get_diffusion_time(
        self, sde, initial_diffusion_time, final_diffusion_time
    ):

        diffusion_time = (
            initial_diffusion_time
            + torch.rand(1) * (final_diffusion_time - initial_diffusion_time)
        )[0]

        sde_time = final_diffusion_time - diffusion_time

        computed_diffusion_time = sde._get_diffusion_time(sde_time)

        torch.testing.assert_close(computed_diffusion_time, diffusion_time)

    def test_sde_g_squared(
        self, sde, noise_parameters, initial_diffusion_time, final_diffusion_time
    ):

        time_array = initial_diffusion_time + torch.rand(1) * (
            final_diffusion_time - initial_diffusion_time
        )

        sigma = VarianceScheduler(noise_parameters).get_sigma(time_array)[0]

        expected_g_squared = (
            2.0
            * sigma**2
            * torch.log(
                torch.tensor(noise_parameters.sigma_max / noise_parameters.sigma_min)
            )
        )

        diffusion_time = time_array[0]

        computed_g_squared = sde._get_diffusion_coefficient_g_squared(diffusion_time)

        torch.testing.assert_close(computed_g_squared, expected_g_squared)

    @pytest.fixture()
    def sde_generator(self, noise_parameters, sampling_parameters, axl_network):
        generator = ExplodingVarianceSDEPositionGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=axl_network,
        )
        return generator

    def test_smoke_sample(
        self,
        sde_generator,
        device,
        number_of_samples,
        number_of_atoms,
        spatial_dimension,
        unit_cell_sample,
    ):
        # Just a smoke test that we can sample without crashing.
        relative_coordinates = sde_generator.sample(
            number_of_samples, device, unit_cell_sample
        )

        assert relative_coordinates.shape == (
            number_of_samples,
            number_of_atoms,
            spatial_dimension,
        )

        assert relative_coordinates.min() >= 0.0
        assert relative_coordinates.max() < 1.0

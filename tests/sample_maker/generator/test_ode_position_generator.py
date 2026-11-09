import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.generator.ode_position_generator import (
    ExplodingVarianceODEAXLGenerator, ODESamplingParameters)
from src.diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from tests.sample_maker.generator.conftest import BaseTestGenerator


@pytest.mark.parametrize("total_time_steps", [2, 5, 10])
@pytest.mark.parametrize("sigma_min_cart", [0.15])
@pytest.mark.parametrize("record_samples", [False, True])
@pytest.mark.parametrize("number_of_samples", [8])
class TestExplodingVarianceODEAXLGenerator(BaseTestGenerator):

    @pytest.fixture()
    def noise_parameters(self, total_time_steps, sigma_min_cart):
        return NoiseParameters(
            total_time_steps=total_time_steps, time_delta=0.0, sigma_min_cart=sigma_min_cart
        )

    @pytest.fixture()
    def sampling_parameters(
        self,
        number_of_atoms,
        spatial_dimension,
        number_of_samples,
        record_samples,
        num_atom_types,
    ):
        sampling_parameters = ODESamplingParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            number_of_samples=number_of_samples,
            record_samples=record_samples,
            num_atom_types=num_atom_types,
        )
        return sampling_parameters

    @pytest.fixture()
    def ode_generator(
        self,
        noise_parameters,
        sampling_parameters,
        axl_network,
    ):
        generator = ExplodingVarianceODEAXLGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=axl_network,
        )

        return generator

    def test_get_ode_prefactor(self, ode_generator, noise_parameters):
        times = NoiseScheduler._get_time_array(noise_parameters)
        sigmas = (
            noise_parameters.sigma_min_cart ** (1.0 - times)
            * noise_parameters.sigma_max_cart**times
        )

        sig_ratio = torch.tensor(
            noise_parameters.sigma_max_cart / noise_parameters.sigma_min_cart
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
    ):
        # Just a smoke test that we can sample without crashing.
        sampled_axl = ode_generator.sample(number_of_samples, device)

        assert sampled_axl.X.shape == (
            number_of_samples,
            number_of_atoms,
            spatial_dimension,
        )

        assert sampled_axl.X.min() >= 0.0
        assert sampled_axl.X.max() < 1.0

    def test_initialize_uses_fixed_lattice_parameters(
        self,
        noise_parameters,
        axl_network,
        device,
        number_of_samples,
        number_of_atoms,
        spatial_dimension,
        num_atom_types,
        cell_dimensions,
        record_samples,
    ):
        sampling_parameters = ODESamplingParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            number_of_samples=number_of_samples,
            num_atom_types=num_atom_types,
            record_samples=record_samples,
            use_fixed_lattice_parameters=True,
            cell_dimensions=cell_dimensions,
        )
        generator = ExplodingVarianceODEAXLGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=axl_network,
        )

        initial_composition = generator.initialize(number_of_samples, device)

        expected_lattice_parameters = sampling_parameters.fixed_lattice_parameters.repeat(number_of_samples, 1)
        torch.testing.assert_close(initial_composition.L, expected_lattice_parameters.to(device))

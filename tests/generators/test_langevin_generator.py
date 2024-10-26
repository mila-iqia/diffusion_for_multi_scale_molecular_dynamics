import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_position_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from src.diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.variance_sampler import \
    ExplodingVarianceSampler
from tests.generators.conftest import BaseTestGenerator


class TestLangevinGenerator(BaseTestGenerator):

    @pytest.fixture(params=[0, 1, 2])
    def number_of_corrector_steps(self, request):
        return request.param

    @pytest.fixture(params=[1, 5, 10])
    def total_time_steps(self, request):
        return request.param

    @pytest.fixture()
    def noise_parameters(self, total_time_steps):
        noise_parameters = NoiseParameters(
            total_time_steps=total_time_steps,
            time_delta=0.1,
            sigma_min=0.15,
            corrector_step_epsilon=0.25,
        )
        return noise_parameters

    @pytest.fixture()
    def sampling_parameters(
        self,
        number_of_atoms,
        spatial_dimension,
        cell_dimensions,
        number_of_samples,
        number_of_corrector_steps,
        unit_cell_size,
    ):
        sampling_parameters = PredictorCorrectorSamplingParameters(
            number_of_corrector_steps=number_of_corrector_steps,
            number_of_atoms=number_of_atoms,
            number_of_samples=number_of_samples,
            cell_dimensions=cell_dimensions,
            spatial_dimension=spatial_dimension,
        )

        return sampling_parameters

    @pytest.fixture()
    def pc_generator(
        self, noise_parameters, sampling_parameters, sigma_normalized_score_network
    ):
        generator = LangevinGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            sigma_normalized_score_network=sigma_normalized_score_network,
        )

        return generator

    def test_smoke_sample(
        self, pc_generator, device, number_of_samples, unit_cell_sample
    ):
        # Just a smoke test that we can sample without crashing.
        pc_generator.sample(number_of_samples, device, unit_cell_sample)

    @pytest.fixture()
    def x_i(self, number_of_samples, number_of_atoms, spatial_dimension, device):
        return map_relative_coordinates_to_unit_cell(
            torch.rand(number_of_samples, number_of_atoms, spatial_dimension)
        ).to(device)

    def test_predictor_step(
        self,
        mocker,
        pc_generator,
        noise_parameters,
        x_i,
        total_time_steps,
        number_of_samples,
        unit_cell_sample,
    ):

        sampler = ExplodingVarianceSampler(noise_parameters)
        noise, _ = sampler.get_all_sampling_parameters()
        sigma_min = noise_parameters.sigma_min
        list_sigma = noise.sigma
        list_time = noise.time
        forces = torch.zeros_like(x_i)

        z = pc_generator._draw_gaussian_sample(number_of_samples).to(x_i)
        mocker.patch.object(pc_generator, "_draw_gaussian_sample", return_value=z)

        for index_i in range(1, total_time_steps + 1):
            computed_sample = pc_generator.predictor_step(
                x_i, index_i, unit_cell_sample, forces
            )

            sigma_i = list_sigma[index_i - 1]
            t_i = list_time[index_i - 1]
            if index_i == 1:
                sigma_im1 = sigma_min
            else:
                sigma_im1 = list_sigma[index_i - 2]

            g2 = sigma_i**2 - sigma_im1**2

            s_i = (
                pc_generator._get_sigma_normalized_scores(
                    x_i, t_i, sigma_i, unit_cell_sample, forces
                )
                / sigma_i
            )

            expected_sample = x_i + g2 * s_i + torch.sqrt(g2) * z

            torch.testing.assert_close(computed_sample, expected_sample)

    def test_corrector_step(
        self,
        mocker,
        pc_generator,
        noise_parameters,
        x_i,
        total_time_steps,
        number_of_samples,
        unit_cell_sample,
    ):

        sampler = ExplodingVarianceSampler(noise_parameters)
        noise, _ = sampler.get_all_sampling_parameters()
        sigma_min = noise_parameters.sigma_min
        epsilon = noise_parameters.corrector_step_epsilon
        list_sigma = noise.sigma
        list_time = noise.time
        sigma_1 = list_sigma[0]
        forces = torch.zeros_like(x_i)

        z = pc_generator._draw_gaussian_sample(number_of_samples).to(x_i)
        mocker.patch.object(pc_generator, "_draw_gaussian_sample", return_value=z)

        for index_i in range(0, total_time_steps):
            computed_sample = pc_generator.corrector_step(
                x_i, index_i, unit_cell_sample, forces
            )

            if index_i == 0:
                sigma_i = sigma_min
                t_i = 0.0
            else:
                sigma_i = list_sigma[index_i - 1]
                t_i = list_time[index_i - 1]

            eps_i = 0.5 * epsilon * sigma_i**2 / sigma_1**2

            s_i = (
                pc_generator._get_sigma_normalized_scores(
                    x_i, t_i, sigma_i, unit_cell_sample, forces
                )
                / sigma_i
            )

            expected_sample = x_i + eps_i * s_i + torch.sqrt(2.0 * eps_i) * z

            torch.testing.assert_close(computed_sample, expected_sample)

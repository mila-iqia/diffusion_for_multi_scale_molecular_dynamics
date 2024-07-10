import pytest
import torch

from crystal_diffusion.generators.langevin_position_generator import \
    AnnealedLangevinDynamicsGenerator
from crystal_diffusion.generators.predictor_corrector_position_generator import \
    PredictorCorrectorSamplingParameters
from crystal_diffusion.models.score_networks import (MLPScoreNetwork,
                                                     MLPScoreNetworkParameters)
from crystal_diffusion.samplers.variance_sampler import (
    ExplodingVarianceSampler, NoiseParameters)
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell

_available_devices = [torch.device('cpu')]
if torch.cuda.is_available():
    _available_devices.append(torch.device('cuda'))


class TestAnnealedLangevinDynamics:
    @pytest.fixture(params=_available_devices)
    def device(self, request):
        return request.param

    @pytest.fixture(params=[0, 1, 2])
    def number_of_corrector_steps(self, request):
        return request.param

    @pytest.fixture(params=[1, 5, 10])
    def total_time_steps(self, request):
        return request.param

    @pytest.fixture(params=[2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture()
    def number_of_atoms(self):
        return 8

    @pytest.fixture()
    def number_of_samples(self):
        return 8

    @pytest.fixture()
    def unit_cell_size(self):
        return 10

    @pytest.fixture()
    def sigma_normalized_score_network(self, number_of_atoms, spatial_dimension):
        hyper_params = MLPScoreNetworkParameters(
            spatial_dimension=spatial_dimension,
            number_of_atoms=number_of_atoms,
            n_hidden_dimensions=3,
            hidden_dimensions_size=16
        )
        return MLPScoreNetwork(hyper_params)

    @pytest.fixture()
    def noise_parameters(self, total_time_steps):
        noise_parameters = NoiseParameters(total_time_steps=total_time_steps,
                                           time_delta=0.1,
                                           sigma_min=0.15,
                                           corrector_step_epsilon=0.25)
        return noise_parameters

    @pytest.fixture()
    def sampling_parameters(self, number_of_atoms, spatial_dimension, number_of_samples,
                            number_of_corrector_steps, unit_cell_size):
        sampling_parameters = PredictorCorrectorSamplingParameters(number_of_corrector_steps=number_of_corrector_steps,
                                                                   number_of_atoms=number_of_atoms,
                                                                   number_of_samples=number_of_samples,
                                                                   cell_dimensions=spatial_dimension * [unit_cell_size],
                                                                   spatial_dimension=spatial_dimension)

        return sampling_parameters

    @pytest.fixture()
    def pc_generator(self, noise_parameters, sampling_parameters, sigma_normalized_score_network):
        generator = AnnealedLangevinDynamicsGenerator(noise_parameters=noise_parameters,
                                                      sampling_parameters=sampling_parameters,
                                                      sigma_normalized_score_network=sigma_normalized_score_network)

        return generator

    @pytest.fixture()
    def unit_cell_sample(self, unit_cell_size, spatial_dimension, number_of_samples):
        return torch.diag(torch.Tensor([unit_cell_size] * spatial_dimension)).repeat(number_of_samples, 1, 1)

    def test_smoke_sample(self, pc_generator, device, number_of_samples, unit_cell_sample):
        # Just a smoke test that we can sample without crashing.
        pc_generator.sample(number_of_samples, device, unit_cell_sample)

    @pytest.fixture()
    def x_i(self, number_of_samples, number_of_atoms, spatial_dimension):
        return map_relative_coordinates_to_unit_cell(torch.rand(number_of_samples, number_of_atoms, spatial_dimension))

    def test_predictor_step(self, mocker, pc_generator, noise_parameters, x_i, total_time_steps, number_of_samples,
                            unit_cell_sample):

        sampler = ExplodingVarianceSampler(noise_parameters)
        noise, _ = sampler.get_all_sampling_parameters()
        sigma_min = noise_parameters.sigma_min
        list_sigma = noise.sigma
        list_time = noise.time
        forces = torch.zeros_like(x_i)

        z = pc_generator._draw_gaussian_sample(number_of_samples)
        mocker.patch.object(pc_generator, "_draw_gaussian_sample", return_value=z)

        for index_i in range(1, total_time_steps + 1):
            computed_sample = pc_generator.predictor_step(x_i, index_i, unit_cell_sample, forces)

            sigma_i = list_sigma[index_i - 1]
            t_i = list_time[index_i - 1]
            if index_i == 1:
                sigma_im1 = sigma_min
            else:
                sigma_im1 = list_sigma[index_i - 2]

            g2 = sigma_i**2 - sigma_im1**2

            s_i = pc_generator._get_sigma_normalized_scores(x_i, t_i, sigma_i, unit_cell_sample, forces) / sigma_i

            expected_sample = x_i + g2 * s_i + torch.sqrt(g2) * z

            torch.testing.assert_close(computed_sample, expected_sample)

    def test_corrector_step(self, mocker, pc_generator, noise_parameters, x_i, total_time_steps, number_of_samples,
                            unit_cell_sample):

        sampler = ExplodingVarianceSampler(noise_parameters)
        noise, _ = sampler.get_all_sampling_parameters()
        sigma_min = noise_parameters.sigma_min
        epsilon = noise_parameters.corrector_step_epsilon
        list_sigma = noise.sigma
        list_time = noise.time
        sigma_1 = list_sigma[0]
        forces = torch.zeros_like(x_i)

        z = pc_generator._draw_gaussian_sample(number_of_samples)
        mocker.patch.object(pc_generator, "_draw_gaussian_sample", return_value=z)

        for index_i in range(0, total_time_steps):
            computed_sample = pc_generator.corrector_step(x_i, index_i, unit_cell_sample, forces)

            if index_i == 0:
                sigma_i = sigma_min
                t_i = 0.
            else:
                sigma_i = list_sigma[index_i - 1]
                t_i = list_time[index_i - 1]

            eps_i = 0.5 * epsilon * sigma_i**2 / sigma_1**2

            s_i = pc_generator._get_sigma_normalized_scores(x_i, t_i, sigma_i, unit_cell_sample, forces) / sigma_i

            expected_sample = x_i + eps_i * s_i + torch.sqrt(2. * eps_i) * z

            torch.testing.assert_close(computed_sample, expected_sample)

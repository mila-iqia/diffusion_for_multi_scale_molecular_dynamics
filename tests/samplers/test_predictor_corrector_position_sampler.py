import pytest
import torch

from crystal_diffusion.models.score_network import (MLPScoreNetwork,
                                                    MLPScoreNetworkParameters)
from crystal_diffusion.samplers.predictor_corrector_position_sampler import (
    AnnealedLangevinDynamicsSampler, PredictorCorrectorPositionSampler)
from crystal_diffusion.samplers.variance_sampler import (
    ExplodingVarianceSampler, NoiseParameters)
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell


class FakePCSampler(PredictorCorrectorPositionSampler):
    """A dummy PC sampler for the purpose of testing."""

    def __init__(
        self,
        number_of_discretization_steps: int,
        number_of_corrector_steps: int,
        spatial_dimension: int,
        initial_sample: torch.Tensor,
    ):
        super().__init__(number_of_discretization_steps, number_of_corrector_steps, spatial_dimension)
        self.initial_sample = initial_sample

    def initialize(self, number_of_samples: int):
        return self.initial_sample

    def predictor_step(self, x_ip1: torch.Tensor, ip1: int, unit_cell: torch.Tensor) -> torch.Tensor:
        return 1.2 * x_ip1 + 3.4 + ip1 / 111.0

    def corrector_step(self, x_i: torch.Tensor, i: int, unit_cell: torch.Tensor) -> torch.Tensor:
        return 0.56 * x_i + 7.89 + i / 117.0


@pytest.mark.parametrize("number_of_samples", [4])
@pytest.mark.parametrize("number_of_atoms", [8])
@pytest.mark.parametrize("spatial_dimension", [2, 3])
@pytest.mark.parametrize("number_of_discretization_steps", [1, 5, 10])
@pytest.mark.parametrize("number_of_corrector_steps", [0, 1, 2])
@pytest.mark.parametrize("unit_cell_size", [10])
class TestPredictorCorrectorPositionSampler:
    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(1234567)

    @pytest.fixture
    def initial_sample(self, number_of_samples, number_of_atoms, spatial_dimension):
        return torch.rand(number_of_samples, number_of_atoms, spatial_dimension)

    @pytest.fixture
    def sampler(
        self, number_of_discretization_steps, number_of_corrector_steps, spatial_dimension, initial_sample
    ):
        sampler = FakePCSampler(
            number_of_discretization_steps, number_of_corrector_steps, spatial_dimension, initial_sample
        )
        return sampler

    @pytest.fixture()
    def unit_cell_sample(self, unit_cell_size, spatial_dimension, number_of_samples):
        return torch.diag(torch.Tensor([unit_cell_size] * spatial_dimension)).repeat(number_of_samples, 1, 1)

    @pytest.fixture
    def expected_samples(
        self,
        sampler,
        initial_sample,
        number_of_discretization_steps,
        number_of_corrector_steps,
        unit_cell_sample,
    ):
        list_i = list(range(number_of_discretization_steps))
        list_i.reverse()
        list_j = list(range(number_of_corrector_steps))

        noisy_sample = map_relative_coordinates_to_unit_cell(initial_sample)
        x_ip1 = noisy_sample
        for i in list_i:
            xi = map_relative_coordinates_to_unit_cell(sampler.predictor_step(x_ip1, i + 1, unit_cell_sample))
            for _ in list_j:
                xi = map_relative_coordinates_to_unit_cell(sampler.corrector_step(xi, i, unit_cell_sample))
            x_ip1 = xi
        return xi

    def test_sample(self, sampler, number_of_samples, expected_samples, unit_cell_sample):
        computed_samples = sampler.sample(number_of_samples, torch.device('cpu'), unit_cell_sample)
        torch.testing.assert_close(expected_samples, computed_samples)


@pytest.mark.parametrize("total_time_steps", [1, 5, 10])
@pytest.mark.parametrize("number_of_corrector_steps", [0, 1, 2])
@pytest.mark.parametrize("spatial_dimension", [2, 3])
@pytest.mark.parametrize("n_hidden_dimensions", [3])
@pytest.mark.parametrize("hidden_dimensions_size", [16])
@pytest.mark.parametrize("number_of_atoms", [8])
@pytest.mark.parametrize("time_delta", [0.1])
@pytest.mark.parametrize("sigma_min", [0.15])
@pytest.mark.parametrize("corrector_step_epsilon", [0.25])
@pytest.mark.parametrize("number_of_samples", [8])
@pytest.mark.parametrize("unit_cell_size", [10])
class TestAnnealedLangevinDynamics:
    @pytest.fixture()
    def sigma_normalized_score_network(
        self, number_of_atoms, spatial_dimension, n_hidden_dimensions, hidden_dimensions_size
    ):
        hyper_params = MLPScoreNetworkParameters(
            spatial_dimension=spatial_dimension,
            number_of_atoms=number_of_atoms,
            n_hidden_dimensions=n_hidden_dimensions,
            hidden_dimensions_size=hidden_dimensions_size
        )
        return MLPScoreNetwork(hyper_params)

    @pytest.fixture()
    def noise_parameters(self, total_time_steps, time_delta, sigma_min, corrector_step_epsilon):
        noise_parameters = NoiseParameters(total_time_steps=total_time_steps,
                                           time_delta=time_delta,
                                           sigma_min=sigma_min,
                                           corrector_step_epsilon=corrector_step_epsilon)
        return noise_parameters

    @pytest.fixture()
    def pc_sampler(self, noise_parameters,
                   number_of_corrector_steps,
                   number_of_atoms,
                   spatial_dimension,
                   sigma_normalized_score_network):
        sampler = AnnealedLangevinDynamicsSampler(noise_parameters=noise_parameters,
                                                  number_of_corrector_steps=number_of_corrector_steps,
                                                  number_of_atoms=number_of_atoms,
                                                  spatial_dimension=spatial_dimension,
                                                  sigma_normalized_score_network=sigma_normalized_score_network)

        return sampler

    @pytest.fixture()
    def unit_cell_sample(self, unit_cell_size, spatial_dimension, number_of_samples):
        return torch.diag(torch.Tensor([unit_cell_size] * spatial_dimension)).repeat(number_of_samples, 1, 1)

    def test_smoke_sample(self, pc_sampler, number_of_samples, unit_cell_sample):
        # Just a smoke test that we can sample without crashing.
        pc_sampler.sample(number_of_samples, torch.device('cpu'), unit_cell_sample)

    @pytest.fixture()
    def x_i(self, number_of_samples, number_of_atoms, spatial_dimension):
        return map_relative_coordinates_to_unit_cell(torch.rand(number_of_samples, number_of_atoms, spatial_dimension))

    def test_predictor_step(self, mocker, pc_sampler, noise_parameters, x_i, total_time_steps, number_of_samples,
                            unit_cell_sample):

        sampler = ExplodingVarianceSampler(noise_parameters)
        noise, _ = sampler.get_all_sampling_parameters()
        sigma_min = noise_parameters.sigma_min
        list_sigma = noise.sigma
        list_time = noise.time

        z = pc_sampler._draw_gaussian_sample(number_of_samples)
        mocker.patch.object(pc_sampler, "_draw_gaussian_sample", return_value=z)

        for index_i in range(1, total_time_steps + 1):
            computed_sample = pc_sampler.predictor_step(x_i, index_i, unit_cell_sample)

            sigma_i = list_sigma[index_i - 1]
            t_i = list_time[index_i - 1]
            if index_i == 1:
                sigma_im1 = sigma_min
            else:
                sigma_im1 = list_sigma[index_i - 2]

            g2 = sigma_i**2 - sigma_im1**2

            s_i = pc_sampler._get_sigma_normalized_scores(x_i, t_i, unit_cell_sample) / sigma_i

            expected_sample = x_i + g2 * s_i + torch.sqrt(g2) * z

            torch.testing.assert_close(computed_sample, expected_sample)

    def test_corrector_step(self, mocker, pc_sampler, noise_parameters, x_i, total_time_steps, number_of_samples,
                            unit_cell_sample):

        sampler = ExplodingVarianceSampler(noise_parameters)
        noise, _ = sampler.get_all_sampling_parameters()
        sigma_min = noise_parameters.sigma_min
        epsilon = noise_parameters.corrector_step_epsilon
        list_sigma = noise.sigma
        list_time = noise.time
        sigma_1 = list_sigma[0]

        z = pc_sampler._draw_gaussian_sample(number_of_samples)
        mocker.patch.object(pc_sampler, "_draw_gaussian_sample", return_value=z)

        for index_i in range(0, total_time_steps):
            computed_sample = pc_sampler.corrector_step(x_i, index_i, unit_cell_sample)

            if index_i == 0:
                sigma_i = sigma_min
                t_i = 0.
            else:
                sigma_i = list_sigma[index_i - 1]
                t_i = list_time[index_i - 1]

            eps_i = 0.5 * epsilon * sigma_i**2 / sigma_1**2

            s_i = pc_sampler._get_sigma_normalized_scores(x_i, t_i, unit_cell_sample) / sigma_i

            expected_sample = x_i + eps_i * s_i + torch.sqrt(2. * eps_i) * z

            torch.testing.assert_close(computed_sample, expected_sample)

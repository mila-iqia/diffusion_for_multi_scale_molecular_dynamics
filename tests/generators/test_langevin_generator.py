import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import (
    class_index_to_onehot, get_probability_at_previous_time_step)
from src.diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.variance_sampler import \
    NoiseScheduler
from tests.generators.conftest import BaseTestGenerator


class TestLangevinGenerator(BaseTestGenerator):

    @pytest.fixture(params=[1, 5, 10])
    def num_atom_types(self, request):
        return request.param

    @pytest.fixture()
    def num_atomic_classes(self, num_atom_types):
        return num_atom_types + 1

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
    def small_epsilon(self):
        return 1e-6

    @pytest.fixture()
    def sampling_parameters(
        self,
        number_of_atoms,
        spatial_dimension,
        cell_dimensions,
        number_of_samples,
        number_of_corrector_steps,
        unit_cell_size,
        num_atom_types,
        small_epsilon,
    ):
        sampling_parameters = PredictorCorrectorSamplingParameters(
            number_of_corrector_steps=number_of_corrector_steps,
            number_of_atoms=number_of_atoms,
            number_of_samples=number_of_samples,
            cell_dimensions=cell_dimensions,
            spatial_dimension=spatial_dimension,
            num_atom_types=num_atom_types,
            small_epsilon=small_epsilon,
        )

        return sampling_parameters

    @pytest.fixture()
    def pc_generator(self, noise_parameters, sampling_parameters, axl_network):
        generator = LangevinGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=axl_network,
        )

        return generator

    def test_smoke_sample(
        self, pc_generator, device, number_of_samples, unit_cell_sample
    ):
        # Just a smoke test that we can sample without crashing.
        pc_generator.sample(number_of_samples, device, unit_cell_sample)

    @pytest.fixture()
    def axl_i(
        self,
        number_of_samples,
        number_of_atoms,
        spatial_dimension,
        num_atomic_classes,
        device,
    ):
        return AXL(
            A=torch.randint(
                0, num_atomic_classes, (number_of_samples, number_of_atoms)
            ).to(device),
            X=map_relative_coordinates_to_unit_cell(
                torch.rand(number_of_samples, number_of_atoms, spatial_dimension)
            ).to(device),
            L=torch.zeros(
                number_of_samples, spatial_dimension * (spatial_dimension - 1)
            ).to(
                device
            ),  # TODO placeholder
        )

    def test_predictor_step_relative_coordinates(
        self,
        mocker,
        pc_generator,
        noise_parameters,
        axl_i,
        total_time_steps,
        number_of_samples,
        unit_cell_sample,
        num_atomic_classes,
        device
    ):

        sampler = NoiseScheduler(noise_parameters, num_classes=num_atomic_classes).to(device)
        noise, _ = sampler.get_all_sampling_parameters()
        sigma_min = noise_parameters.sigma_min
        list_sigma = noise.sigma
        list_time = noise.time
        forces = torch.zeros_like(axl_i.X)

        z = pc_generator._draw_gaussian_sample(number_of_samples).to(axl_i.X)
        mocker.patch.object(pc_generator, "_draw_gaussian_sample", return_value=z)

        for index_i in range(1, total_time_steps + 1):
            computed_sample = pc_generator.predictor_step(
                axl_i, index_i, unit_cell_sample, forces
            )

            sigma_i = list_sigma[index_i - 1]
            t_i = list_time[index_i - 1]
            if index_i == 1:
                sigma_im1 = sigma_min
            else:
                sigma_im1 = list_sigma[index_i - 2]

            g2 = sigma_i**2 - sigma_im1**2

            s_i = (
                pc_generator._get_model_predictions(
                    axl_i, t_i, sigma_i, unit_cell_sample, forces
                ).X
                / sigma_i
            )

            expected_coordinates = axl_i.X + g2 * s_i + torch.sqrt(g2) * z
            expected_coordinates = map_relative_coordinates_to_unit_cell(
                expected_coordinates
            )

            torch.testing.assert_close(computed_sample.X, expected_coordinates)

    def test_predictor_step_atom_types(
        self,
        mocker,
        pc_generator,
        noise_parameters,
        axl_i,
        total_time_steps,
        number_of_samples,
        unit_cell_sample,
        num_atomic_classes,
        small_epsilon,
        number_of_atoms,
        device
    ):

        sampler = NoiseScheduler(noise_parameters, num_classes=num_atomic_classes).to(device)
        noise, _ = sampler.get_all_sampling_parameters()
        list_sigma = noise.sigma
        list_time = noise.time
        list_q_matrices = noise.q_matrix
        list_q_bar_matrices = noise.q_bar_matrix
        list_q_bar_tm1_matrices = noise.q_bar_tm1_matrix
        forces = torch.zeros_like(axl_i.X)

        u = pc_generator._draw_gumbel_sample(number_of_samples).to(
            device=axl_i.A.device
        )
        mocker.patch.object(pc_generator, "_draw_gumbel_sample", return_value=u)

        for index_i in range(1, total_time_steps + 1):
            computed_sample = pc_generator.predictor_step(
                axl_i, index_i, unit_cell_sample, forces
            )

            sigma_i = list_sigma[index_i - 1]
            t_i = list_time[index_i - 1]

            p_ao_given_at_i = pc_generator._get_model_predictions(
                axl_i, t_i, sigma_i, unit_cell_sample, forces
            ).A

            onehot_at = class_index_to_onehot(axl_i.A, num_classes=num_atomic_classes)
            q_matrices = list_q_matrices[index_i - 1]
            q_bar_matrices = list_q_bar_matrices[index_i - 1]
            q_bar_tm1_matrices = list_q_bar_tm1_matrices[index_i - 1]

            p_atm1_given_at = get_probability_at_previous_time_step(
                probability_at_zeroth_timestep=p_ao_given_at_i,
                one_hot_probability_at_current_timestep=onehot_at,
                q_matrices=q_matrices,
                q_bar_matrices=q_bar_matrices,
                q_bar_tm1_matrices=q_bar_tm1_matrices,
                small_epsilon=small_epsilon,
                probability_at_zeroth_timestep_are_logits=True,
            )
            gumbel_distribution = torch.log(p_atm1_given_at) + u

            expected_atom_types = torch.argmax(gumbel_distribution, dim=-1)

            torch.testing.assert_close(computed_sample.A, expected_atom_types)

    def test_corrector_step(
        self,
        mocker,
        pc_generator,
        noise_parameters,
        axl_i,
        total_time_steps,
        number_of_samples,
        unit_cell_sample,
        num_atomic_classes,
    ):

        sampler = NoiseScheduler(noise_parameters, num_classes=num_atomic_classes)
        noise, _ = sampler.get_all_sampling_parameters()
        sigma_min = noise_parameters.sigma_min
        epsilon = noise_parameters.corrector_step_epsilon
        list_sigma = noise.sigma
        list_time = noise.time
        sigma_1 = list_sigma[0]
        forces = torch.zeros_like(axl_i.X)

        z = pc_generator._draw_gaussian_sample(number_of_samples).to(axl_i.X)
        mocker.patch.object(pc_generator, "_draw_gaussian_sample", return_value=z)

        for index_i in range(0, total_time_steps):
            computed_sample = pc_generator.corrector_step(
                axl_i, index_i, unit_cell_sample, forces
            )

            if index_i == 0:
                sigma_i = sigma_min
                t_i = 0.0
            else:
                sigma_i = list_sigma[index_i - 1]
                t_i = list_time[index_i - 1]

            eps_i = 0.5 * epsilon * sigma_i**2 / sigma_1**2

            s_i = (
                pc_generator._get_model_predictions(
                    axl_i, t_i, sigma_i, unit_cell_sample, forces
                ).X
                / sigma_i
            )

            expected_coordinates = axl_i.X + eps_i * s_i + torch.sqrt(2.0 * eps_i) * z
            expected_coordinates = map_relative_coordinates_to_unit_cell(
                expected_coordinates
            )

            torch.testing.assert_close(computed_sample.X, expected_coordinates)
            assert torch.all(computed_sample.A == axl_i.A)

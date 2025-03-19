import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.adaptive_corrector import \
    AdaptiveCorrectorGenerator
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from src.diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from tests.generators.test_langevin_generator import TestLangevinGenerator


class TestAdaptiveCorrectorGenerator(TestLangevinGenerator):

    @pytest.fixture()
    def noise_parameters(self, total_time_steps):
        noise_parameters = NoiseParameters(
            total_time_steps=total_time_steps,
            time_delta=0.1,
            sigma_min=0.15,
            corrector_r=0.15,
        )
        return noise_parameters

    @pytest.fixture()
    def pc_generator(self, noise_parameters, sampling_parameters, axl_network):
        # override the base class
        generator = AdaptiveCorrectorGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=axl_network,
        )

        return generator

    def test_predictor_step_relative_coordinates_and_lattice(
        self,
        mocker,
        pc_generator,
        noise_parameters,
        axl_i,
        total_time_steps,
        number_of_samples,
        num_atomic_classes,
        device,
    ):
        # override the base class
        forces = torch.zeros_like(axl_i.X)

        for index_i in range(1, total_time_steps + 1):
            computed_sample = pc_generator.predictor_step(axl_i, index_i, forces)

            expected_coordinates = axl_i.X
            expected_coordinates = map_relative_coordinates_to_unit_cell(
                expected_coordinates
            )
            # this is almost trivial - the coordinates should not change in a predictor step
            torch.testing.assert_close(computed_sample.X, expected_coordinates)

            expected_lattice = axl_i.L
            torch.testing.assert_close(computed_sample.L, expected_lattice)

    @pytest.mark.parametrize("corrector_r", [0.1, 0.5, 1.2])
    def test_corrector_step(
        self,
        mocker,
        corrector_r,
        pc_generator,
        noise_parameters,
        axl_i,
        total_time_steps,
        number_of_samples,
        num_atomic_classes,
        number_of_atoms,
        spatial_dimension
    ):
        pc_generator.corrector_r = corrector_r
        sampler = NoiseScheduler(noise_parameters, num_classes=num_atomic_classes)
        noise, _ = sampler.get_all_sampling_parameters()
        sigma_min = noise_parameters.sigma_min
        list_sigma = noise.sigma
        list_time = noise.time
        forces = torch.zeros_like(axl_i.X)

        z_coordinates = pc_generator._draw_coordinates_gaussian_sample(
            number_of_samples
        ).to(axl_i.X)
        mocker.patch.object(
            pc_generator,
            "_draw_coordinates_gaussian_sample",
            return_value=z_coordinates,
        )

        z_lattice = pc_generator._draw_lattice_gaussian_sample(number_of_samples).to(
            axl_i.L
        )
        mocker.patch.object(
            pc_generator, "_draw_lattice_gaussian_sample", return_value=z_lattice
        )

        z_coordinates_norm = torch.sqrt(
            (z_coordinates**2).sum(dim=-1)
        ).mean()  # norm of z
        z_lattice_norm = torch.sqrt(
            (z_lattice**2).sum(dim=-1)
        ).mean()  # norm of z averaged over lattice parameters

        for index_i in range(0, total_time_steps):
            computed_sample = pc_generator.corrector_step(axl_i, index_i, forces)

            if index_i == 0:
                sigma_i = sigma_min
                t_i = 0.0
            else:
                sigma_i = list_sigma[index_i - 1]
                t_i = list_time[index_i - 1]

            model_predictions = pc_generator._get_model_predictions(
                axl_i, t_i, sigma_i, forces
            )

            # test coordinates update
            s_i_coordinates = model_predictions.X / sigma_i

            s_i_coordinates_norm = torch.sqrt(
                (s_i_coordinates**2).sum(dim=-1).sum(dim=-1)
            ).mean(dim=-1)
            # \epsilon_i = 2 \left(r \frac{||z||_2}{||s(x_i, t_i)||_2}\right)^2
            eps_i_coordinates = (
                2
                * (
                    corrector_r
                    * z_coordinates_norm
                    / s_i_coordinates_norm.clip(min=pc_generator.small_epsilon)
                )
                ** 2
            )
            eps_i_coordinates = eps_i_coordinates.view(1, 1, 1)

            expected_coordinates = (
                axl_i.X
                + eps_i_coordinates * s_i_coordinates
                + torch.sqrt(2.0 * eps_i_coordinates) * z_coordinates
            )
            expected_coordinates = map_relative_coordinates_to_unit_cell(
                expected_coordinates
            )

            torch.testing.assert_close(computed_sample.X, expected_coordinates)

            # test lattice parameters update
            sigma_i_for_lattice = sigma_i / (number_of_atoms ** (1 / spatial_dimension))
            s_i_lattice = model_predictions.L / sigma_i_for_lattice

            s_i_lattice_norm = torch.sqrt((s_i_lattice**2).sum(dim=-1)).mean()
            # \epsilon_i = 2 \left(r \frac{||z||_2}{||s(x_i, t_i)||_2}\right)^2
            eps_i_lattice = (
                2
                * (
                    corrector_r
                    * z_lattice_norm
                    / s_i_lattice_norm.clip(min=pc_generator.small_epsilon)
                )
                ** 2
            )
            eps_i_lattice = eps_i_lattice.view(-1, 1)

            expected_lattice = (
                axl_i.L
                + eps_i_lattice * s_i_lattice
                + torch.sqrt(2.0 * eps_i_lattice) * z_lattice
            )

            torch.testing.assert_close(computed_sample.L, expected_lattice)

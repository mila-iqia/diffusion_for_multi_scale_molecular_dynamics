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
            sigma_min_cart=0.15,
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
        spatial_dimension,
    ):
        pc_generator.corrector_r = corrector_r
        sampler = NoiseScheduler(noise_parameters, num_classes=num_atomic_classes)
        noise, _ = sampler.get_all_sampling_parameters()
        sigma_min_cart = noise_parameters.sigma_min_cart
        list_sigma = noise.sigma
        list_time = noise.time
        forces = torch.zeros_like(axl_i.X)
        lattice_diagonals = axl_i.L[:, :spatial_dimension]

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

        # Match the generator's exact norm computation to avoid fp discrepancies amplified by /L.
        # The generator uses torch.linalg.norm(...).mean() / sigma, not norm(x/sigma).mean().
        z_coordinates_norm = torch.linalg.norm(z_coordinates, dim=-1).mean().view(1, 1, 1)
        z_lattice_norm = torch.linalg.norm(z_lattice, dim=-1).mean().view(1, 1)

        for index_i in range(0, total_time_steps):
            computed_sample = pc_generator.corrector_step(axl_i, index_i, forces)

            if index_i == 0:
                sigma_i = sigma_min_cart
                t_i = 0.0
            else:
                sigma_i = list_sigma[index_i - 1]
                t_i = list_time[index_i - 1]

            model_predictions = pc_generator._get_model_predictions(
                axl_i, t_i, sigma_i, forces
            )

            # test coordinates update — match _generic_corrector_step_size exactly
            sigma_score_norm_coordinates = (
                torch.linalg.norm(model_predictions.X, dim=[-2, -1]).mean() / sigma_i
            ).view(1, 1, 1)
            eps_i_coordinates = (
                2 * (corrector_r * z_coordinates_norm
                     / sigma_score_norm_coordinates.clip(min=pc_generator.small_epsilon))**2
            )

            expected_coordinates = map_relative_coordinates_to_unit_cell(
                axl_i.X
                + eps_i_coordinates / lattice_diagonals[:, None, :] * model_predictions.X / sigma_i
                + torch.sqrt(2.0 * eps_i_coordinates) / lattice_diagonals[:, None, :] * z_coordinates
            )

            torch.testing.assert_close(computed_sample.X, expected_coordinates)

            # TODO: Unsure if there should be a dependence on the number of atoms.
            number_of_atoms = axl_i.X.shape[1]
            sigma_n_i_corrector = sigma_i / (number_of_atoms ** (1.0 / spatial_dimension))

            # test lattice parameters update — match _generic_corrector_step_size exactly
            sigma_score_norm_lattice = (
                torch.linalg.norm(model_predictions.L, dim=-1).mean() / sigma_n_i_corrector
            ).view(1, 1)
            eps_i_lattice = (
                2 * (corrector_r * z_lattice_norm / sigma_score_norm_lattice.clip(min=pc_generator.small_epsilon)) ** 2
            )

            expected_lattice = (
                axl_i.L
                + eps_i_lattice * model_predictions.L / sigma_n_i_corrector
                + torch.sqrt(2.0 * eps_i_lattice) * z_lattice
            )

            torch.testing.assert_close(computed_sample.L, expected_lattice)

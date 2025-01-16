import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from diffusion_for_multi_scale_molecular_dynamics.regularizers.consistency_regularizer import (
    ConsistencyRegularizer, ConsistencyRegularizerParameters)
from diffusion_for_multi_scale_molecular_dynamics.score.wrapped_gaussian_score import \
    get_sigma_normalized_score
from tests.regularizers.conftest import BaseTestRegularizer


class TestConsistencyRegularizer(BaseTestRegularizer):

    @pytest.fixture()
    def device(self):
        # Regularizer currently does not work with device other than CPU. fix if needed.
        return torch.device('cpu')

    @pytest.fixture()
    def maximum_number_of_steps(self):
        return 5

    @pytest.fixture()
    def total_time_steps(
        self,
    ):
        return 10

    @pytest.fixture()
    def noise_parameters(self, sigma_min, sigma_max, total_time_steps):
        return NoiseParameters(
            total_time_steps=total_time_steps, sigma_min=sigma_min, sigma_max=sigma_max
        )

    @pytest.fixture()
    def sampling_parameters(
        self, num_atom_types, number_of_atoms, batch_size, cell_dimensions
    ):
        return PredictorCorrectorSamplingParameters(
            number_of_corrector_steps=0,
            num_atom_types=num_atom_types,
            cell_dimensions=list(cell_dimensions.numpy()),
            number_of_atoms=number_of_atoms,
            number_of_samples=batch_size,
        )

    @pytest.fixture()
    def regularizer_parameters(
        self, maximum_number_of_steps, noise_parameters, sampling_parameters
    ):
        return ConsistencyRegularizerParameters(
            maximum_number_of_steps=maximum_number_of_steps,
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
        )

    @pytest.fixture()
    def noise(self, noise_parameters, num_atom_types):
        sampler = NoiseScheduler(noise_parameters, num_classes=num_atom_types + 1)
        noise, _ = sampler.get_all_sampling_parameters()
        return noise

    @pytest.fixture()
    def regularizer(self, regularizer_parameters, device):
        return ConsistencyRegularizer(regularizer_parameters).to(device)

    def test_get_augmented_batch_for_fixed_time(
        self, regularizer, augmented_batch, batch_size
    ):

        composition = augmented_batch[NOISY_AXL_COMPOSITION]
        unit_cells = augmented_batch[UNIT_CELL]
        times = augmented_batch[TIME].squeeze(-1)
        sigmas = augmented_batch[NOISE].squeeze(-1)

        for time, sigma in zip(times, sigmas):
            new_batch = regularizer.get_augmented_batch_for_fixed_time(
                composition, unit_cells, time, sigma
            )
            torch.testing.assert_close(
                new_batch[TIME], time * torch.ones(batch_size, 1)
            )
            torch.testing.assert_close(
                new_batch[NOISE], sigma * torch.ones(batch_size, 1)
            )

            for key in [NOISY_AXL_COMPOSITION, UNIT_CELL, CARTESIAN_FORCES]:
                torch.testing.assert_close(new_batch[key], augmented_batch[key])

    def test_generate_starting_composition(
        self, regularizer, augmented_batch, batch_size
    ):

        original_composition = augmented_batch[NOISY_AXL_COMPOSITION]
        for batch_index in range(batch_size):
            new_composition = regularizer.generate_starting_composition(
                original_composition, batch_index
            )
            torch.testing.assert_close(new_composition.A, original_composition.A)
            torch.testing.assert_close(new_composition.L, original_composition.L)
            assert new_composition.X.shape == original_composition.X.shape

    def test_get_score_target(self, regularizer, relative_coordinates):

        start_composition = AXL(A=0.0, X=relative_coordinates, L=0.0)
        end_composition = AXL(A=0.0, X=torch.zeros_like(relative_coordinates), L=0.0)
        end_sigma, start_sigma = torch.rand(2).sort().values

        computed_target = regularizer.get_score_target(
            start_composition, end_composition, start_sigma, end_sigma
        )

        sigma_eff = torch.sqrt(start_sigma**2 - end_sigma**2)
        expected_target = (
            start_sigma
            / (sigma_eff)
            * get_sigma_normalized_score(
                relative_coordinates,
                sigma_eff * torch.ones_like(relative_coordinates),
                kmax=regularizer.kmax_target_score,
            )
        )

        torch.testing.assert_close(computed_target, expected_target)

    def test_get_partial_trajectory_start_and_end(
        self, regularizer, noise, maximum_number_of_steps
    ):
        number_of_time_steps = len(noise.time)
        small_time_noise = 1e-6 * (torch.rand(number_of_time_steps) - 0.5)

        #   Predictor(x_{I_START}, I_START)  < ---- >  start_time ~  noise_scheduler.time[I_START - 1]
        for expected_start_time_index, delta in zip(
            range(1, number_of_time_steps + 1), small_time_noise
        ):
            expected_end_time_index = max(
                expected_start_time_index - maximum_number_of_steps, 0
            )

            start_time = noise.time[expected_start_time_index - 1] + delta

            computed_start_time_index, computed_end_time_index, end_time, end_sigma = (
                regularizer.get_partial_trajectory_start_and_end(start_time, noise)
            )

            assert computed_start_time_index == expected_start_time_index
            assert computed_end_time_index == expected_end_time_index

            if expected_end_time_index == 0:
                assert end_sigma == 0.0
                assert end_time == 0.0
            else:
                assert end_sigma == noise.sigma[expected_end_time_index - 1]
                assert end_time == noise.time[expected_end_time_index - 1]

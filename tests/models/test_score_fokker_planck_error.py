from typing import Callable

import einops
import pytest
import torch

from crystal_diffusion.models.normalized_score_fokker_planck_error import \
    NormalizedScoreFokkerPlanckError
from crystal_diffusion.models.score_networks.egnn_score_network import \
    EGNNScoreNetworkParameters
from crystal_diffusion.models.score_networks.score_network_factory import \
    create_score_network
from crystal_diffusion.namespace import (NOISE, NOISY_RELATIVE_COORDINATES,
                                         TIME, UNIT_CELL)
from crystal_diffusion.samplers.exploding_variance import ExplodingVariance
from crystal_diffusion.samplers.variance_sampler import NoiseParameters


def get_finite_difference_time_derivative(
    tensor_function: Callable,
    relative_coordinates: torch.Tensor,
    times: torch.Tensor,
    unit_cells: torch.Tensor,
    epsilon: float = 1.0e-8,
):
    """Compute the finite difference of a tensor function with respect to time."""
    h = epsilon * torch.ones_like(times)
    f_hp = tensor_function(relative_coordinates, times + h, unit_cells)
    f_hm = tensor_function(relative_coordinates, times - h, unit_cells)

    batch_size, natoms, spatial_dimension = relative_coordinates.shape
    denominator = einops.repeat(2 * h, "b 1 -> b n s", n=natoms, s=spatial_dimension)
    time_derivative = (f_hp - f_hm) / denominator
    return time_derivative


def get_finite_difference_gradient(
    scalar_function: Callable,
    relative_coordinates: torch.Tensor,
    times: torch.Tensor,
    unit_cells: torch.Tensor,
    epsilon: float = 1.0e-6,
):
    """Compute the gradient of a scalar function using finite difference."""
    batch_size, natoms, spatial_dimension = relative_coordinates.shape

    x = relative_coordinates

    gradient = torch.zeros_like(relative_coordinates)
    for atom_idx in range(natoms):
        for space_idx in range(spatial_dimension):
            dx = torch.zeros_like(relative_coordinates)
            dx[:, atom_idx, space_idx] = epsilon

            f_p = scalar_function(x + dx, times, unit_cells)
            f_m = scalar_function(x - dx, times, unit_cells)

            gradient[:, atom_idx, space_idx] = (f_p - f_m) / (2.0 * epsilon)

    return gradient


def get_finite_difference_divergence(
    tensor_function: Callable,
    relative_coordinates: torch.Tensor,
    times: torch.Tensor,
    unit_cells: torch.Tensor,
    epsilon: float = 1.0e-8,
):
    """Compute the finite difference divergence of a tensor function."""
    batch_size, natoms, spatial_dimension = relative_coordinates.shape

    x = relative_coordinates
    finite_difference_divergence = torch.zeros(batch_size)

    for atom_idx in range(natoms):
        for space_idx in range(spatial_dimension):
            dx = torch.zeros_like(relative_coordinates)
            dx[:, atom_idx, space_idx] = epsilon
            vec_hp = tensor_function(x + dx, times, unit_cells)
            vec_hm = tensor_function(x - dx, times, unit_cells)
            div_contribution = (
                vec_hp[:, atom_idx, space_idx] - vec_hm[:, atom_idx, space_idx]
            ) / (2.0 * epsilon)
            finite_difference_divergence += div_contribution

    return finite_difference_divergence


class TestScoreFokkerPlanckError:
    @pytest.fixture(scope="class", autouse=True)
    def set_default_type_to_float64(self):
        torch.set_default_dtype(torch.float64)
        yield
        # this returns the default type to float32 at the end of all tests in this class in order
        # to not affect other tests.
        torch.set_default_dtype(torch.float32)

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(23423423)

    @pytest.fixture
    def batch_size(self):
        return 5

    @pytest.fixture
    def spatial_dimension(self):
        return 3

    @pytest.fixture(params=[True, False])
    def inference_mode(self, request):
        return request.param

    @pytest.fixture(params=[2, 4])
    def number_of_atoms(self, request):
        return request.param

    @pytest.fixture
    def relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension):
        return torch.rand(batch_size, number_of_atoms, spatial_dimension)

    @pytest.fixture
    def times(self, batch_size):
        times = torch.rand(batch_size, 1)
        return times

    @pytest.fixture
    def unit_cells(self, batch_size, spatial_dimension):
        return torch.rand(batch_size, spatial_dimension, spatial_dimension)

    @pytest.fixture()
    def score_network_parameters(self, number_of_atoms, spatial_dimension):
        # Let's test with a "real" model to identify any snag in the diff engine.
        score_network_parameters = EGNNScoreNetworkParameters(
            spatial_dimension=spatial_dimension,
            message_n_hidden_dimensions=2,
            node_n_hidden_dimensions=2,
            coordinate_n_hidden_dimensions=2,
            n_layers=2,
        )
        return score_network_parameters

    @pytest.fixture()
    def noise_parameters(self):
        return NoiseParameters(total_time_steps=10, sigma_min=0.1, sigma_max=0.5)

    @pytest.fixture()
    def batch(self, relative_coordinates, times, unit_cells, noise_parameters):
        return {
            NOISY_RELATIVE_COORDINATES: relative_coordinates,
            TIME: times,
            NOISE: ExplodingVariance(noise_parameters).get_sigma(times),
            UNIT_CELL: unit_cells,
        }

    @pytest.fixture()
    def sigma_normalized_score_network(self, score_network_parameters, inference_mode):
        score_network = create_score_network(score_network_parameters)
        if inference_mode:
            for parameter in score_network.parameters():
                parameter.requires_grad_(False)

        return score_network

    @pytest.fixture()
    def expected_normalized_scores(self, sigma_normalized_score_network, batch):
        return sigma_normalized_score_network(batch)

    @pytest.fixture
    def normalized_score_fokker_planck_error(
        self, sigma_normalized_score_network, noise_parameters
    ):
        return NormalizedScoreFokkerPlanckError(
            sigma_normalized_score_network, noise_parameters
        )

    def test_normalized_scores_function(
        self, expected_normalized_scores, normalized_score_fokker_planck_error, batch
    ):
        computed_normalized_scores = (
            normalized_score_fokker_planck_error._normalized_scores_function(
                relative_coordinates=batch[NOISY_RELATIVE_COORDINATES],
                times=batch[TIME],
                unit_cells=batch[UNIT_CELL],
            )
        )

        torch.testing.assert_allclose(
            expected_normalized_scores, computed_normalized_scores
        )

    def test_normalized_scores_square_norm_function(
        self, expected_normalized_scores, normalized_score_fokker_planck_error, batch
    ):
        flat_scores = einops.rearrange(
            expected_normalized_scores, "batch natoms space -> batch (natoms space)"
        )

        expected_squared_norms = (flat_scores**2).sum(dim=1)

        computed_squared_norms = normalized_score_fokker_planck_error._normalized_scores_square_norm_function(
            relative_coordinates=batch[NOISY_RELATIVE_COORDINATES],
            times=batch[TIME],
            unit_cells=batch[UNIT_CELL],
        )

        torch.testing.assert_allclose(expected_squared_norms, computed_squared_norms)

    def test_get_dn_dt(
        self,
        normalized_score_fokker_planck_error,
        relative_coordinates,
        times,
        unit_cells,
    ):
        finite_difference_dn_dt = get_finite_difference_time_derivative(
            normalized_score_fokker_planck_error._normalized_scores_function,
            relative_coordinates,
            times,
            unit_cells,
        )

        computed_dn_dt = normalized_score_fokker_planck_error._get_dn_dt(
            relative_coordinates, times, unit_cells
        )
        torch.testing.assert_close(computed_dn_dt, finite_difference_dn_dt)

    def test_divergence_function(
        self,
        normalized_score_fokker_planck_error,
        relative_coordinates,
        times,
        unit_cells,
    ):
        finite_difference_divergence = get_finite_difference_divergence(
            normalized_score_fokker_planck_error._normalized_scores_function,
            relative_coordinates,
            times,
            unit_cells,
        )

        computed_divergence = normalized_score_fokker_planck_error._divergence_function(
            relative_coordinates, times, unit_cells
        )

        torch.testing.assert_close(computed_divergence, finite_difference_divergence)

    def test_get_gradient(
        self,
        normalized_score_fokker_planck_error,
        relative_coordinates,
        times,
        unit_cells,
    ):
        for callable in [
            normalized_score_fokker_planck_error._divergence_function,
            normalized_score_fokker_planck_error._normalized_scores_square_norm_function,
        ]:
            computed_grads = normalized_score_fokker_planck_error._get_gradient(
                callable, relative_coordinates, times, unit_cells
            )
            finite_difference_grads = get_finite_difference_gradient(
                callable, relative_coordinates, times, unit_cells
            )

            torch.testing.assert_close(computed_grads, finite_difference_grads)

    def test_get_normalized_score_fokker_planck_error(
        self,
        normalized_score_fokker_planck_error,
        relative_coordinates,
        times,
        unit_cells,
    ):
        errors1 = normalized_score_fokker_planck_error.get_normalized_score_fokker_planck_error(
            relative_coordinates, times, unit_cells
        )

        errors2 = normalized_score_fokker_planck_error.get_normalized_score_fokker_planck_error_by_iterating_over_batch(
            relative_coordinates, times, unit_cells
        )

        torch.testing.assert_allclose(errors1, errors2)

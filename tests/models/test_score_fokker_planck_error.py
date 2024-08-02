import einops
import pytest
import torch

from crystal_diffusion.models.score_fokker_planck_error import \
    ScoreFokkerPlanckError
from crystal_diffusion.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetworkParameters, TargetScoreBasedAnalyticalScoreNetwork)
from crystal_diffusion.namespace import (NOISE, NOISY_RELATIVE_COORDINATES,
                                         TIME, UNIT_CELL)
from crystal_diffusion.samplers.exploding_variance import ExplodingVariance
from crystal_diffusion.samplers.variance_sampler import NoiseParameters


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
        return 4

    @pytest.fixture
    def kmax(self):
        # kmax has to be fairly large for the comparison test between the analytical score and the target based
        # analytical score to pass.
        return 8

    @pytest.fixture(params=[1, 2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture(params=[4, 8])
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
    def score_network_parameters(self, number_of_atoms, spatial_dimension, kmax):
        equilibrium_relative_coordinates = torch.rand(
            number_of_atoms, spatial_dimension
        )
        hyper_params = AnalyticalScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            kmax=kmax,
            equilibrium_relative_coordinates=equilibrium_relative_coordinates,
            variance_parameter=0.1,
            use_permutation_invariance=False,
        )
        return hyper_params

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
    def sigma_normalized_score_network(self, score_network_parameters):
        return TargetScoreBasedAnalyticalScoreNetwork(score_network_parameters)

    @pytest.fixture
    def expected_scores(self, sigma_normalized_score_network, batch):
        sigma_normalized_scores = sigma_normalized_score_network(batch)
        _, natoms, spatial_dimension = sigma_normalized_scores.shape
        sigmas = batch[NOISE]
        scores = sigma_normalized_scores / einops.repeat(
            sigmas, "b 1 -> b n s", n=natoms, s=spatial_dimension
        )
        return scores

    @pytest.fixture
    def score_fokker_planck_error(
        self, sigma_normalized_score_network, noise_parameters
    ):
        return ScoreFokkerPlanckError(sigma_normalized_score_network, noise_parameters)

    def test_get_score(
        self,
        score_fokker_planck_error,
        relative_coordinates,
        times,
        unit_cells,
        expected_scores,
    ):
        computed_scores = score_fokker_planck_error._get_scores(
            relative_coordinates, times, unit_cells
        )

        torch.testing.assert_close(computed_scores, expected_scores)

    def test_get_score_time_derivative(
        self,
        score_fokker_planck_error,
        relative_coordinates,
        times,
        unit_cells,
        expected_scores,
    ):
        # Finite difference approximation
        h = 1e-8 * torch.ones_like(times)
        scores_hp = score_fokker_planck_error._get_scores(
            relative_coordinates, times + h, unit_cells
        )
        scores_hm = score_fokker_planck_error._get_scores(
            relative_coordinates, times - h, unit_cells
        )

        batch_size, natoms, spatial_dimension = scores_hp.shape
        denominator = einops.repeat(
            2 * h, "b 1 -> b n s", n=natoms, s=spatial_dimension
        )
        expected_score_time_derivative = (scores_hp - scores_hm) / denominator

        t = torch.tensor(times, requires_grad=True)
        computed_score_time_derivative = (
            score_fokker_planck_error._get_score_time_derivative(
                relative_coordinates, t, unit_cells
            )
        )
        torch.testing.assert_close(
            computed_score_time_derivative, expected_score_time_derivative
        )

    def test_get_score_divergence(
        self,
        score_fokker_planck_error,
        relative_coordinates,
        times,
        unit_cells,
        expected_scores,
    ):
        # Finite difference approximation
        epsilon = 1e-8

        batch_size, natoms, spatial_dimension = relative_coordinates.shape

        expected_score_divergence = torch.zeros(batch_size)

        for atom_idx in range(natoms):
            for space_idx in range(spatial_dimension):
                dx = torch.zeros_like(relative_coordinates)
                dx[:, atom_idx, space_idx] = epsilon
                scores_hp = score_fokker_planck_error._get_scores(
                    relative_coordinates + dx, times, unit_cells
                )
                scores_hm = score_fokker_planck_error._get_scores(
                    relative_coordinates - dx, times, unit_cells
                )
                dscore = (
                    scores_hp[:, atom_idx, space_idx]
                    - scores_hm[:, atom_idx, space_idx]
                )

                expected_score_divergence += dscore / (2.0 * epsilon)

        x = torch.tensor(relative_coordinates, requires_grad=True)
        computed_score_divergence = score_fokker_planck_error._get_score_divergence(
            x, times, unit_cells
        )

        torch.testing.assert_close(computed_score_divergence, expected_score_divergence)

    def test_get_scores_square_norm(
        self, score_fokker_planck_error, relative_coordinates, times, unit_cells
    ):
        scores = score_fokker_planck_error._get_scores(
            relative_coordinates, times, unit_cells
        )

        batch_size, natoms, spatial_dimension = relative_coordinates.shape

        expected_score_norms = torch.zeros(batch_size)
        for atom_idx in range(natoms):
            for space_idx in range(spatial_dimension):
                expected_score_norms += scores[:, atom_idx, space_idx] ** 2

        computed_score_norms = score_fokker_planck_error._get_scores_square_norm(
            relative_coordinates, times, unit_cells
        )

        torch.testing.assert_close(computed_score_norms, expected_score_norms)

    def test_get_gradient_term(
        self, score_fokker_planck_error, relative_coordinates, times, unit_cells
    ):
        x = torch.tensor(relative_coordinates, requires_grad=True)

        epsilon = 1.0e-6
        batch_size, natoms, spatial_dimension = relative_coordinates.shape

        expected_gradient_term = torch.zeros_like(relative_coordinates)
        for atom_idx in range(natoms):
            for space_idx in range(spatial_dimension):
                dx = torch.zeros_like(relative_coordinates)
                dx[:, atom_idx, space_idx] = epsilon

                ns_p = score_fokker_planck_error._get_score_divergence(
                    x + dx, times, unit_cells
                )
                s2_p = score_fokker_planck_error._get_scores_square_norm(
                    x + dx, times, unit_cells
                )
                ns_m = score_fokker_planck_error._get_score_divergence(
                    x - dx, times, unit_cells
                )
                s2_m = score_fokker_planck_error._get_scores_square_norm(
                    x - dx, times, unit_cells
                )

                expected_gradient_term[:, atom_idx, space_idx] = (
                    ns_p + s2_p - ns_m - s2_m
                ) / (2.0 * epsilon)

        computed_gradient_term = score_fokker_planck_error._get_gradient_term(
            x, times, unit_cells
        )

        torch.testing.assert_close(expected_gradient_term, computed_gradient_term)

    def test_get_score_fokker_planck_error(
        self, score_fokker_planck_error, relative_coordinates, times, unit_cells
    ):
        errors = score_fokker_planck_error.get_score_fokker_planck_error(
            relative_coordinates, times, unit_cells
        )
        # since we are using an analytical score, which is exact, the FP equation should be exactly satisfied.
        torch.testing.assert_close(errors, torch.zeros_like(errors))

import itertools

import pytest
import torch

from crystal_diffusion.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from crystal_diffusion.namespace import (NOISE, NOISY_RELATIVE_COORDINATES,
                                         TIME, UNIT_CELL)


def factorial(n):

    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)


class TestAnalyticalScoreNetwork:

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(23423423)

    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def kmax(self):
        return 1

    @pytest.fixture(params=[1, 2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture(params=[1, 2])
    def number_of_atoms(self, request):
        return request.param

    @pytest.fixture
    def equilibrium_relative_coordinates(self, number_of_atoms, spatial_dimension):
        return torch.rand(number_of_atoms, spatial_dimension)

    @pytest.fixture
    def variance_parameter(self):
        # Make the spring constants pretty large so that the displacements will be small
        inverse_variance = float(1000 * torch.rand(1))
        return 1. / inverse_variance

    @pytest.fixture()
    def batch(self, batch_size, number_of_atoms, spatial_dimension):
        relative_coordinates = torch.rand(batch_size, number_of_atoms, spatial_dimension)
        times = torch.rand(batch_size, 1)
        noises = torch.rand(batch_size, 1)
        unit_cell = torch.rand(batch_size, spatial_dimension, spatial_dimension)
        return {NOISY_RELATIVE_COORDINATES: relative_coordinates, TIME: times, NOISE: noises, UNIT_CELL: unit_cell}

    @pytest.fixture()
    def score_network_parameters(self, number_of_atoms, spatial_dimension, kmax,
                                 equilibrium_relative_coordinates, variance_parameter, use_permutation_invariance):
        hyper_params = AnalyticalScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            kmax=kmax,
            equilibrium_relative_coordinates=equilibrium_relative_coordinates,
            variance_parameter=variance_parameter,
            use_permutation_invariance=use_permutation_invariance)
        return hyper_params

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        return AnalyticalScoreNetwork(score_network_parameters)

    def test_all_translations(self, kmax):
        computed_translations = AnalyticalScoreNetwork._get_all_translations(kmax)
        expected_translations = torch.tensor(list(range(-kmax, kmax + 1)))
        torch.testing.assert_close(expected_translations, computed_translations)

    def test_get_all_equilibrium_permutations(self, number_of_atoms, spatial_dimension,
                                              equilibrium_relative_coordinates):
        expected_permutations = []

        for permutation_indices in itertools.permutations(range(number_of_atoms)):
            expected_permutations.append(equilibrium_relative_coordinates[list(permutation_indices)])

        expected_permutations = torch.stack(expected_permutations)

        computed_permutations = (
            AnalyticalScoreNetwork._get_all_equilibrium_permutations(equilibrium_relative_coordinates))

        assert computed_permutations.shape == (factorial(number_of_atoms), number_of_atoms, spatial_dimension)

        torch.testing.assert_close(expected_permutations, computed_permutations)

    @pytest.mark.parametrize('use_permutation_invariance', [False])
    def test_compute_unnormalized_log_probability(self, equilibrium_relative_coordinates, variance_parameter,
                                                  kmax, batch, score_network):
        sigmas = batch[NOISE]  # dimension: [batch_size, 1]
        xt = batch[NOISY_RELATIVE_COORDINATES]
        computed_log_prob = score_network._compute_unnormalized_log_probability(sigmas,
                                                                                xt,
                                                                                equilibrium_relative_coordinates)

        batch_size = sigmas.shape[0]

        expected_log_prob = torch.zeros(batch_size)
        for batch_idx in range(batch_size):
            sigma = sigmas[batch_idx, 0]

            for i in range(score_network.natoms):
                for alpha in range(score_network.spatial_dimension):

                    eq_coordinate = equilibrium_relative_coordinates[i, alpha]
                    coordinate = xt[batch_idx, i, alpha]

                    sum_on_k = torch.tensor(0.)
                    for k in range(-kmax, kmax + 1):
                        exponent = -0.5 * (coordinate - eq_coordinate - k)**2 / (sigma**2 + variance_parameter)
                        sum_on_k += torch.exp(exponent)

                    expected_log_prob[batch_idx] += torch.log(sum_on_k)

        torch.testing.assert_close(expected_log_prob, computed_log_prob)

    @pytest.mark.parametrize('number_of_atoms, use_permutation_invariance', [(1, False), (1, True),
                                                                             (2, False), (2, True), (8, False)])
    def test_analytical_score_network(self, score_network, batch, batch_size, number_of_atoms, spatial_dimension):
        normalized_scores = score_network.forward(batch)

        assert normalized_scores.shape == (batch_size, number_of_atoms, spatial_dimension)

import itertools

import einops
import pytest
import torch

from crystal_diffusion.models.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from crystal_diffusion.namespace import (NOISE, NOISY_RELATIVE_COORDINATES,
                                         TIME, UNIT_CELL)
from tests.fake_data_utils import find_aligning_permutation


def factorial(n):

    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)


@pytest.mark.parametrize("spatial_dimension", [1, 2, 3])
@pytest.mark.parametrize("number_of_atoms", [1, 2])
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

    @pytest.fixture
    def equilibrium_relative_coordinates(self, number_of_atoms, spatial_dimension):
        return torch.rand(number_of_atoms, spatial_dimension)

    @pytest.fixture
    def inverse_covariance(self, number_of_atoms, spatial_dimension):

        combined_dimension = number_of_atoms * spatial_dimension

        # extract a random orthogonal matrix
        random_matrix = torch.rand(combined_dimension, combined_dimension)
        orthogonal_matrix, _, _ = torch.svd(random_matrix)

        # Make the spring constants pretty large so that the displacements will be small
        spring_constants = 5. + 5. * torch.rand(combined_dimension)

        flat_inverse_covariance = orthogonal_matrix @ (torch.diag(spring_constants) @ orthogonal_matrix.T)

        return einops.rearrange(flat_inverse_covariance, '(n1 d1) (n2 d2) -> n1 d1 n2 d2',
                                n1=number_of_atoms, d1=spatial_dimension, n2=number_of_atoms, d2=spatial_dimension)

    @pytest.fixture()
    def batch(self, batch_size, number_of_atoms, spatial_dimension):
        relative_coordinates = torch.rand(batch_size, number_of_atoms, spatial_dimension)
        times = torch.rand(batch_size, 1)
        noises = torch.rand(batch_size, 1)
        unit_cell = torch.rand(batch_size, spatial_dimension, spatial_dimension)
        return {NOISY_RELATIVE_COORDINATES: relative_coordinates, TIME: times, NOISE: noises, UNIT_CELL: unit_cell}

    @pytest.fixture()
    def score_network_parameters(self, number_of_atoms, spatial_dimension, kmax,
                                 equilibrium_relative_coordinates, inverse_covariance):
        hyper_params = AnalyticalScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            kmax=kmax,
            equilibrium_relative_coordinates=equilibrium_relative_coordinates,
            inverse_covariance=inverse_covariance)
        return hyper_params

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        return AnalyticalScoreNetwork(score_network_parameters)

    def test_all_translations(self, kmax, number_of_atoms, spatial_dimension):

        flat_dim = number_of_atoms * spatial_dimension

        computed_translations = AnalyticalScoreNetwork._get_all_translations(kmax, flat_dim)

        expected_number_of_translations = (2 * kmax + 1) ** flat_dim
        assert computed_translations.shape == (expected_number_of_translations, flat_dim)

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

    def test_get_effective_inverse_covariance_matrices(self, number_of_atoms, spatial_dimension,
                                                       batch, inverse_covariance, score_network):
        sigmas = batch[NOISE]
        flat_dim = number_of_atoms * spatial_dimension

        beta_phi = einops.rearrange(inverse_covariance, 'n1 d1 n2 d2 -> (n1 d1) (n2 d2)',
                                    n1=number_of_atoms, d1=spatial_dimension,
                                    n2=number_of_atoms, d2=spatial_dimension)

        list_matrices = []
        for sigma in sigmas:
            matrix = torch.linalg.inv(torch.eye(flat_dim) + sigma**2 * beta_phi) @ beta_phi
            list_matrices.append(matrix)

        expected_matrices = torch.stack(list_matrices)

        computed_matrices = score_network._get_effective_inverse_covariance_matrices(sigmas)

        torch.testing.assert_close(computed_matrices, expected_matrices)

    def test_get_all_flat_offsets(self, score_network):
        permutations_x0 = score_network.permutations_x0
        translations_k = score_network.translations_k

        list_offsets = []
        for perm in permutations_x0:
            for tran in translations_k:
                offset = perm + tran
                list_offsets.append(offset)

        expected_all_offsets = torch.stack(list_offsets)

        computed_all_offsets = score_network._get_all_flat_offsets(permutations_x0, translations_k)

        perm = find_aligning_permutation(expected_all_offsets, computed_all_offsets)

        torch.testing.assert_close(expected_all_offsets, computed_all_offsets[perm])

    def test_analytical_score_network(self, score_network, batch):
        _ = score_network.forward(batch)

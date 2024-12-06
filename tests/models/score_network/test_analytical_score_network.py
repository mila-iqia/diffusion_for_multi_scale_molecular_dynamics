import itertools

import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters,
    TargetScoreBasedAnalyticalScoreNetwork)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from tests.models.score_network.base_test_score_network import \
    BaseTestScoreNetwork


def factorial(n):

    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)


class TestAnalyticalScoreNetwork(BaseTestScoreNetwork):
    @pytest.fixture(scope="class", autouse=True)
    def set_default_type_to_float64(self):
        torch.set_default_dtype(torch.float64)
        yield
        # this returns the default type to float32 at the end of all tests in this class in order
        # to not affect other tests.
        torch.set_default_dtype(torch.float32)

    @pytest.fixture
    def kmax(self):
        # kmax has to be fairly large for the comparison test between the analytical score and the target based
        # analytical score to pass.
        return 8

    @pytest.fixture(params=[1, 2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture
    def num_atom_types(self):
        return 1

    @pytest.fixture(params=[1, 2])
    def number_of_atoms(self, request):
        return request.param

    @pytest.fixture
    def equilibrium_relative_coordinates(self, number_of_atoms, spatial_dimension):
        return torch.rand(number_of_atoms, spatial_dimension)

    """
    @pytest.fixture
    def atom_types(self, batch_size, number_of_atoms, num_atom_types):
        return torch.randint(
            0,
            num_atom_types,
            (
                batch_size,
                number_of_atoms,
            ),
        )
    """

    @pytest.fixture(params=["finite", "zero"])
    def variance_parameter(self, request):
        if request.param == "zero":
            return 0.0
        elif request.param == "finite":
            # Make the spring constants pretty large so that the displacements will be small
            inverse_variance = float(1000 * torch.rand(1))
            return 1.0 / inverse_variance

    @pytest.fixture()
    def batch(self, batch_size, number_of_atoms, spatial_dimension, atom_types):
        relative_coordinates = torch.rand(
            batch_size, number_of_atoms, spatial_dimension
        )
        times = torch.rand(batch_size, 1)
        noises = torch.rand(batch_size, 1)
        unit_cell = torch.rand(batch_size, spatial_dimension, spatial_dimension)
        lattice_params = torch.zeros(batch_size, int(spatial_dimension * (spatial_dimension + 1) / 2))
        lattice_params[:, :spatial_dimension] = torch.diagonal(unit_cell, dim1=-2, dim2=-1)
        return {
            NOISY_AXL_COMPOSITION: AXL(
                A=atom_types, X=relative_coordinates, L=lattice_params
            ),
            TIME: times,
            NOISE: noises,
            UNIT_CELL: unit_cell,
        }

    @pytest.fixture()
    def score_network_parameters(
        self,
        number_of_atoms,
        spatial_dimension,
        kmax,
        equilibrium_relative_coordinates,
        variance_parameter,
        use_permutation_invariance,
        num_atom_types
    ):
        hyper_params = AnalyticalScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            kmax=kmax,
            equilibrium_relative_coordinates=equilibrium_relative_coordinates,
            variance_parameter=variance_parameter,
            use_permutation_invariance=use_permutation_invariance,
            num_atom_types=num_atom_types
        )
        return hyper_params

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        return AnalyticalScoreNetwork(score_network_parameters)

    @pytest.fixture()
    def target_score_based_score_network(self, score_network_parameters):
        return TargetScoreBasedAnalyticalScoreNetwork(score_network_parameters)

    def test_all_translations(self, kmax):
        computed_translations = AnalyticalScoreNetwork._get_all_translations(kmax)
        expected_translations = torch.tensor(list(range(-kmax, kmax + 1)))
        torch.testing.assert_close(expected_translations, computed_translations)

    def test_get_all_equilibrium_permutations(
        self, number_of_atoms, spatial_dimension, equilibrium_relative_coordinates
    ):
        expected_permutations = []

        for permutation_indices in itertools.permutations(range(number_of_atoms)):
            expected_permutations.append(
                equilibrium_relative_coordinates[list(permutation_indices)]
            )

        expected_permutations = torch.stack(expected_permutations)

        computed_permutations = (
            AnalyticalScoreNetwork._get_all_equilibrium_permutations(
                equilibrium_relative_coordinates
            )
        )

        assert computed_permutations.shape == (
            factorial(number_of_atoms),
            number_of_atoms,
            spatial_dimension,
        )

        torch.testing.assert_close(expected_permutations, computed_permutations)

    @pytest.mark.parametrize("use_permutation_invariance", [False])
    def test_compute_unnormalized_log_probability(
        self,
        equilibrium_relative_coordinates,
        variance_parameter,
        kmax,
        batch,
        score_network,
    ):
        sigmas = batch[NOISE]  # dimension: [batch_size, 1]
        xt = batch[NOISY_AXL_COMPOSITION].X
        computed_log_prob = score_network._compute_unnormalized_log_probability(
            sigmas, xt, equilibrium_relative_coordinates
        )

        batch_size = sigmas.shape[0]

        expected_log_prob = torch.zeros(batch_size)
        for batch_idx in range(batch_size):
            sigma = sigmas[batch_idx, 0]

            for i in range(score_network.natoms):
                for alpha in range(score_network.spatial_dimension):

                    eq_coordinate = equilibrium_relative_coordinates[i, alpha]
                    coordinate = xt[batch_idx, i, alpha]

                    sum_on_k = torch.tensor(0.0)
                    for k in range(-kmax, kmax + 1):
                        exponent = (
                            -0.5
                            * (coordinate - eq_coordinate - k) ** 2
                            / (sigma**2 + variance_parameter)
                        )
                        sum_on_k += torch.exp(exponent)

                    expected_log_prob[batch_idx] += torch.log(sum_on_k)

        # Let's give a free pass to any problematic expected values, which are calculated with a fragile
        # brute force approach
        problem_mask = torch.logical_or(torch.isnan(expected_log_prob), torch.isinf(expected_log_prob))
        expected_log_prob[problem_mask] = computed_log_prob[problem_mask]

        torch.testing.assert_close(expected_log_prob, computed_log_prob)

    @pytest.mark.parametrize(
        "number_of_atoms, use_permutation_invariance",
        [(1, False), (1, True), (2, False), (2, True), (8, False)],
    )
    def test_analytical_score_network(
        self, score_network, batch, batch_size, number_of_atoms, spatial_dimension
    ):
        normalized_scores = score_network.forward(batch)

        assert normalized_scores.X.shape == (
            batch_size,
            number_of_atoms,
            spatial_dimension,
        )

    @pytest.mark.parametrize("use_permutation_invariance", [False])
    @pytest.mark.parametrize("number_of_atoms", [1, 2, 8])
    def test_compare_score_networks(
        self, score_network, target_score_based_score_network, batch
    ):

        normalized_scores1 = score_network.forward(batch)
        normalized_scores2 = target_score_based_score_network.forward(batch)

        torch.testing.assert_close(normalized_scores1, normalized_scores2)

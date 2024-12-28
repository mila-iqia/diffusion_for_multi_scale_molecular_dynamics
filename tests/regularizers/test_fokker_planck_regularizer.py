import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.regularizers.fokker_planck_regularizer import (
    FokkerPlanckRegularizer, RegularizerParameters)
from tests.regularizers.differentiable_score_network import (
    DifferentiableScoreNetwork, DifferentiableScoreNetworkParameters)


class TestFokkerPlanckRegularizer:

    @pytest.fixture(scope="class", autouse=True)
    def set_default_type_to_float64(self):
        torch.set_default_dtype(torch.float64)
        yield
        # this returns the default type to float32 at the end of all tests in this class in order
        # to not affect other tests.
        torch.set_default_dtype(torch.float32)

    @pytest.fixture(scope="class", autouse=True)
    def set_seed(self):
        """Set the random seed."""
        torch.manual_seed(34534234)

    @pytest.fixture()
    def sigma_min(self):
        return 0.001

    @pytest.fixture()
    def sigma_max(self):
        return 0.2

    @pytest.fixture()
    def number_of_hte_terms(self):
        return 4

    @pytest.fixture()
    def number_of_atoms(self):
        return 4

    @pytest.fixture()
    def num_atom_types(self):
        return 1

    @pytest.fixture()
    def spatial_dimension(self):
        return 3

    @pytest.fixture()
    def batch_size(self):
        return 16

    @pytest.fixture()
    def relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension):
        return torch.rand(batch_size, number_of_atoms, spatial_dimension)

    @pytest.fixture()
    def times(self, batch_size):
        return torch.rand(batch_size, 1)

    @pytest.fixture()
    def atom_types(self, batch_size, number_of_atoms):
        return torch.zeros(batch_size, number_of_atoms, dtype=torch.int64)

    @pytest.fixture()
    def unit_cells(self, batch_size, spatial_dimension):
        acell = 5.0
        unit_cells = torch.diag(acell * torch.ones(spatial_dimension)).repeat(
            batch_size, 1, 1
        )
        return unit_cells

    @pytest.fixture()
    def score_parameters(
        self, number_of_atoms, num_atom_types, spatial_dimension, sigma_min, sigma_max
    ):
        score_parameters = DifferentiableScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            num_atom_types=num_atom_types,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        return score_parameters

    @pytest.fixture()
    def score_network(self, score_parameters):
        return DifferentiableScoreNetwork(score_parameters)

    @pytest.fixture()
    def regularizer_parameters(
        self, batch_size, number_of_hte_terms, sigma_min, sigma_max
    ):
        return RegularizerParameters(
            regularizer_lambda_weight=1.0,
            batch_size=batch_size,
            number_of_hte_terms=number_of_hte_terms,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    @pytest.fixture()
    def regularizer(self, regularizer_parameters):
        return FokkerPlanckRegularizer(regularizer_parameters)

    @pytest.fixture()
    def score_function(self, regularizer, score_network, atom_types, unit_cells):
        score_function = regularizer._create_score_function(
            score_network, atom_types, unit_cells
        )
        return score_function

    def test_score_function(
        self, score_network, score_function, relative_coordinates, times
    ):
        computed_scores = score_function(relative_coordinates, times)
        expected_scores = score_network._score_function(relative_coordinates, times)
        torch.testing.assert_allclose(computed_scores, expected_scores)

    def test_create_rademacher_random_variables(
        self,
        regularizer,
        number_of_hte_terms,
        batch_size,
        number_of_atoms,
        spatial_dimension,
    ):
        z = regularizer._create_rademacher_random_variables(
            batch_size, number_of_atoms, spatial_dimension
        )
        assert z.shape == (
            number_of_hte_terms,
            batch_size,
            number_of_atoms,
            spatial_dimension,
        )
        torch.testing.assert_close(z.unique(), torch.tensor([-1.0, 1.0]))

    def test_get_hte_laplacian_term(
        self, score_network, regularizer, score_function, relative_coordinates, times
    ):
        batch_size, num_atoms, spatial_dimension = relative_coordinates.shape

        def score_function_x(relative_coordinates):
            return score_function(relative_coordinates, times)

        rademacher_z = regularizer._create_rademacher_random_variables(
            batch_size=batch_size,
            num_atoms=num_atoms,
            spatial_dimension=spatial_dimension,
        )

        exact_hessian = score_network._space_hessian_function(
            relative_coordinates, times
        )

        for z in rademacher_z:
            computed_laplacian = regularizer.get_hte_laplacian_term(
                score_function_x, relative_coordinates, z
            )
            hz = einops.einsum(
                exact_hessian,
                z,
                "batch ni si nj sj nk sk, batch nk sk -> batch ni si nj sj",
            )
            expected_laplacian = einops.einsum(
                z, hz, "batch nj sj, batch ni si nj sj -> batch ni si"
            )
            torch.testing.assert_allclose(computed_laplacian, expected_laplacian)

    def test_compute_residual_components(
        self,
        regularizer,
        score_network,
        relative_coordinates,
        times,
        atom_types,
        unit_cells,
    ):
        batch = regularizer._create_batch(
            relative_coordinates, times, atom_types, unit_cells
        )
        rademacher_z = regularizer._create_rademacher_random_variables(
            *relative_coordinates.shape
        )

        (
            scores,
            scores_time_derivative,
            scores_divergence_scores,
            approximate_scores_laplacian,
        ) = regularizer.compute_residual_components(score_network, batch, rademacher_z)

        expected_scores = score_network._score_function(relative_coordinates, times)
        expected_scores_time_derivatives = score_network._time_derivative_function(
            relative_coordinates, times
        )

        space_jacobian = score_network._space_jacobian_function(
            relative_coordinates, times
        )
        expected_scores_divergence_scores = einops.einsum(
            space_jacobian,
            expected_scores,
            "batch ni si nj sj, batch nj sj -> batch ni si",
        )

        expected_approximate_scores_laplacian = torch.zeros_like(expected_scores)

        exact_hessian = score_network._space_hessian_function(
            relative_coordinates, times
        )

        n_hte = len(rademacher_z)
        for z in rademacher_z:
            hz = einops.einsum(
                exact_hessian,
                z,
                "batch ni si nj sj nk sk, batch nk sk -> batch ni si nj sj",
            )
            zhz = einops.einsum(z, hz, "batch nj sj, batch ni si nj sj -> batch ni si")
            expected_approximate_scores_laplacian += zhz / n_hte

        torch.testing.assert_allclose(scores, expected_scores)
        torch.testing.assert_allclose(
            scores_time_derivative, expected_scores_time_derivatives
        )
        torch.testing.assert_allclose(
            scores_divergence_scores, expected_scores_divergence_scores
        )
        torch.testing.assert_allclose(
            approximate_scores_laplacian, expected_approximate_scores_laplacian
        )

    def test_compute_score_fokker_planck_residuals(
        self,
        regularizer,
        score_network,
        relative_coordinates,
        times,
        atom_types,
        unit_cells,
    ):
        batch = regularizer._create_batch(
            relative_coordinates, times, atom_types, unit_cells
        )
        residuals = regularizer.compute_score_fokker_planck_residuals(
            score_network, batch
        )

        assert residuals.shape == relative_coordinates.shape

    def test_compute_weighted_regularizer_loss(
        self,
        regularizer,
        score_network,
        relative_coordinates,
        times,
        atom_types,
        unit_cells,
    ):
        # Smoke test that the method runs.
        _ = regularizer.compute_weighted_regularizer_loss(
            score_network, times, atom_types, unit_cells
        )

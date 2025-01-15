import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.regularizers.fokker_planck_regularizer import (
    FokkerPlanckRegularizer, FokkerPlanckRegularizerParameters)
from tests.regularizers.conftest import BaseTestRegularizer
from tests.regularizers.differentiable_score_network import \
    DifferentiableScoreNetworkParameters


class TestFokkerPlanckRegularizer(BaseTestRegularizer):

    @pytest.fixture()
    def device(self):
        # Regularizer currently does not work with device other than CPU. fix if needed.
        return torch.device('cpu')

    @pytest.fixture(scope="class", autouse=True)
    def set_default_type_to_float64(self):
        torch.set_default_dtype(torch.float64)
        yield
        # this returns the default type to float32 at the end of all tests in this class in order
        # to not affect other tests.
        torch.set_default_dtype(torch.float32)

    @pytest.fixture()
    def number_of_hte_terms(self):
        return 4

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
    def regularizer_parameters(
        self, batch_size, number_of_hte_terms, sigma_min, sigma_max
    ):
        parameters = FokkerPlanckRegularizerParameters(
            regularizer_lambda_weight=1.0,
            number_of_hte_terms=number_of_hte_terms,
            use_hte_approximation=True,
            batch_size=batch_size,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        return parameters

    @pytest.fixture()
    def regularizer(self, regularizer_parameters, device):
        return FokkerPlanckRegularizer(regularizer_parameters, device)

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

    def test_get_exact_laplacian_term(
        self, score_network, regularizer, score_function, relative_coordinates, times
    ):
        def score_function_x(relative_coordinates):
            return score_function(relative_coordinates, times)

        computed_laplacian = regularizer.get_exact_laplacian(
            score_function_x, relative_coordinates
        )

        exact_hessian = score_network._space_hessian_function(
            relative_coordinates, times
        )
        expected_laplacian = einops.einsum(
            exact_hessian, "batch ni si nj sj nj sj -> batch ni si"
        )

        torch.testing.assert_allclose(computed_laplacian, expected_laplacian)

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

        (
            scores,
            scores_time_derivative,
            scores_divergence_scores,
            scores_laplacian,
        ) = regularizer.compute_residual_components(score_network, batch)

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

        torch.testing.assert_allclose(scores, expected_scores)
        torch.testing.assert_allclose(
            scores_time_derivative, expected_scores_time_derivatives
        )
        torch.testing.assert_allclose(
            scores_divergence_scores, expected_scores_divergence_scores
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

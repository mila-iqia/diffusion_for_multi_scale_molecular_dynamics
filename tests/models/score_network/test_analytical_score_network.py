import itertools

import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.utils.symmetry_utils import \
    factorial
from tests.models.score_network.base_test_score_network import \
    BaseTestScoreNetwork


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
        list_x = torch.rand(number_of_atoms, spatial_dimension)
        return [list(x.numpy()) for x in list_x]

    @pytest.fixture(params=[True, False])
    def use_permutation_invariance(self, request):
        return request.param

    @pytest.fixture(params=[0.01, 0.1, 0.5])
    def sigma_d(self, request):
        return request.param

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
        sigma_d,
        use_permutation_invariance,
        num_atom_types,
    ):
        hyper_params = AnalyticalScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            kmax=kmax,
            equilibrium_relative_coordinates=equilibrium_relative_coordinates,
            sigma_d=sigma_d,
            use_permutation_invariance=use_permutation_invariance,
            num_atom_types=num_atom_types,
        )
        return hyper_params

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        return AnalyticalScoreNetwork(score_network_parameters)

    def test_all_translations(self, kmax):
        computed_translations = AnalyticalScoreNetwork._get_all_translations(kmax)
        expected_translations = torch.tensor(list(range(-kmax, kmax + 1)))
        torch.testing.assert_close(expected_translations, computed_translations)

    def test_get_all_equilibrium_permutations(
        self, number_of_atoms, spatial_dimension, equilibrium_relative_coordinates
    ):

        eq_rel_coords = torch.tensor(equilibrium_relative_coordinates)
        expected_permutations = []

        for permutation_indices in itertools.permutations(range(number_of_atoms)):
            expected_permutations.append(
                eq_rel_coords[list(permutation_indices)]
            )

        expected_permutations = torch.stack(expected_permutations)

        computed_permutations = (
            AnalyticalScoreNetwork._get_all_equilibrium_permutations(
                eq_rel_coords)
        )

        assert computed_permutations.shape == (
            factorial(number_of_atoms),
            number_of_atoms,
            spatial_dimension,
        )

        torch.testing.assert_close(expected_permutations, computed_permutations)

    def compute_log_wrapped_gaussian_for_testing(
        self,
        relative_coordinates,
        equilibrium_relative_coordinates,
        sigmas,
        sigma_d,
        kmax,
    ):
        """Compute the log of a Wrapped Gaussian, for testing purposes."""
        batch_size, natoms, spatial_dimension = relative_coordinates.shape

        assert sigmas.shape == (
            batch_size,
            1,
        ), "Unexpected shape for the sigmas tensor."

        assert equilibrium_relative_coordinates.shape == (
            natoms,
            spatial_dimension,
        ), "A single equilibrium configuration should be used."

        nd = natoms * spatial_dimension

        list_translations = torch.arange(-kmax, kmax + 1)
        nt = len(list_translations)

        # Recast various spatial arrays to the correct dimensions to combine them,
        # in dimensions [batch, nd, number_of_translations]
        effective_variance = einops.repeat(
            sigmas**2 + sigma_d**2, "batch 1 -> batch nd t", t=nt, nd=nd
        )

        x = einops.repeat(
            relative_coordinates, "batch natoms space -> batch (natoms space) t", t=nt
        )

        x0 = einops.repeat(
            equilibrium_relative_coordinates,
            "natoms space -> batch (natoms space) t",
            batch=batch_size,
            t=nt,
        )

        translations = einops.repeat(
            list_translations, "t -> batch nd t", batch=batch_size, nd=nd
        )

        exponent = -0.5 * (x - x0 - translations) ** 2 / effective_variance
        # logsumexp on lattice translation vectors, then sum on spatial indices
        unnormalized_log_prob = torch.logsumexp(exponent, dim=-1, keepdim=False).sum(
            dim=1
        )

        sigma2 = effective_variance[
            :, :, 0
        ]  # We shouldn't sum the normalization term on k.
        normalization_term = torch.sum(
            0.5 * torch.log(2.0 * torch.pi * sigma2), dim=[-1]
        )

        log_wrapped_gaussian = unnormalized_log_prob - normalization_term

        return log_wrapped_gaussian

    @pytest.fixture()
    def all_equilibrium_permutations(self, score_network):
        return score_network.all_x0

    @pytest.fixture()
    def expected_wrapped_gaussians(
        self, batch, all_equilibrium_permutations, sigma_d, kmax
    ):
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        sigmas = batch[NOISE]

        list_log_w = []
        for x0 in all_equilibrium_permutations:
            log_w = self.compute_log_wrapped_gaussian_for_testing(
                relative_coordinates, x0, sigmas, sigma_d=sigma_d, kmax=kmax
            )
            list_log_w.append(log_w)

        expected_wrapped_gaussians = torch.stack(list_log_w)
        return expected_wrapped_gaussians

    def test_log_wrapped_gaussians_computation(
        self, expected_wrapped_gaussians, score_network, batch
    ):
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        sigmas = batch[NOISE]
        batch_size, natoms, space_dimensions = relative_coordinates.shape

        sigmas_t = einops.repeat(
            sigmas,
            "batch 1 -> batch natoms space",
            natoms=natoms,
            space=space_dimensions,
        )
        computed_wrapped_gaussians, _ = (
            score_network.get_log_wrapped_gaussians_and_normalized_scores_centered_on_equilibrium_positions(
                relative_coordinates, sigmas_t
            )
        )

        torch.testing.assert_close(
            expected_wrapped_gaussians, computed_wrapped_gaussians
        )

    def compute_sigma_normalized_scores_by_autograd_for_testing(
        self,
        relative_coordinates,
        equilibrium_relative_coordinates,
        sigmas,
        sigma_d,
        kmax,
    ):
        """Compute scores by autograd, for testing."""
        batch_size, natoms, spatial_dimension = relative_coordinates.shape
        xt = relative_coordinates.clone()
        xt.requires_grad_(True)

        log_w = self.compute_log_wrapped_gaussian_for_testing(
            xt, equilibrium_relative_coordinates, sigmas, sigma_d=sigma_d, kmax=kmax
        )
        grad_outputs = [torch.ones_like(log_w)]

        scores = torch.autograd.grad(
            outputs=[log_w], inputs=[xt], grad_outputs=grad_outputs
        )[0]

        # We actually want sigma x score.
        broadcast_sigmas = einops.repeat(
            sigmas,
            "batch 1 -> batch natoms space",
            natoms=natoms,
            space=spatial_dimension,
        )

        sigma_normalized_scores = broadcast_sigmas * scores
        return sigma_normalized_scores

    @pytest.fixture()
    def expected_sigma_normalized_scores(
        self, batch, all_equilibrium_permutations, sigma_d, kmax
    ):
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        sigmas = batch[NOISE]

        list_sigma_normalized_scores = []
        for x0 in all_equilibrium_permutations:
            sigma_scores = self.compute_sigma_normalized_scores_by_autograd_for_testing(
                relative_coordinates, x0, sigmas, sigma_d=sigma_d, kmax=kmax
            )
            list_sigma_normalized_scores.append(sigma_scores)
        sigma_normalized_scores = torch.stack(list_sigma_normalized_scores)
        return sigma_normalized_scores

    def test_normalized_score_computation(
        self, expected_sigma_normalized_scores, score_network, batch
    ):
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        sigmas = batch[NOISE]
        batch_size, natoms, space_dimensions = relative_coordinates.shape

        sigmas_t = einops.repeat(
            sigmas,
            "batch 1 -> batch natoms space",
            natoms=natoms,
            space=space_dimensions,
        )
        _, computed_normalized_scores = (
            score_network.get_log_wrapped_gaussians_and_normalized_scores_centered_on_equilibrium_positions(
                relative_coordinates, sigmas_t
            )
        )

        torch.testing.assert_close(
            expected_sigma_normalized_scores, computed_normalized_scores
        )

    @pytest.mark.parametrize(
        "number_of_atoms, use_permutation_invariance",
        [(1, False), (1, True), (2, False), (2, True), (8, False)],
    )
    def test_analytical_score_network_shapes(
        self,
        score_network,
        batch,
        batch_size,
        number_of_atoms,
        num_atom_types,
        spatial_dimension,
    ):
        model_axl = score_network.forward(batch)

        normalized_scores = model_axl.X
        atom_type_preds = model_axl.A

        assert normalized_scores.shape == (
            batch_size,
            number_of_atoms,
            spatial_dimension,
        )
        assert atom_type_preds.shape == (
            batch_size,
            number_of_atoms,
            num_atom_types + 1,
        )

    @pytest.mark.parametrize(
        "number_of_atoms, use_permutation_invariance",
        [(1, False), (1, True), (2, False), (2, True), (8, False)],
    )
    def test_analytical_score_network(
        self, score_network, batch, sigma_d, kmax, equilibrium_relative_coordinates
    ):
        model_axl = score_network.forward(batch)

        computed_normalized_scores = model_axl.X

        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        batch_size, natoms, spatial_dimension = relative_coordinates.shape
        sigmas = batch[NOISE]

        list_log_w = []
        list_s = []
        for x0 in score_network.all_x0:
            log_w = self.compute_log_wrapped_gaussian_for_testing(
                relative_coordinates, x0, sigmas, sigma_d, kmax
            )
            list_log_w.append(log_w)

            s = self.compute_sigma_normalized_scores_by_autograd_for_testing(
                relative_coordinates, x0, sigmas, sigma_d, kmax
            )
            list_s.append(s)

        list_log_w = torch.stack(list_log_w)
        list_s = torch.stack(list_s)

        list_weights = einops.repeat(
            torch.softmax(list_log_w, dim=0),
            "n batch -> n batch natoms space",
            natoms=natoms,
            space=spatial_dimension,
        )

        expected_normalized_scores = torch.sum(list_weights * list_s, dim=0)

        torch.testing.assert_close(
            expected_normalized_scores, computed_normalized_scores
        )

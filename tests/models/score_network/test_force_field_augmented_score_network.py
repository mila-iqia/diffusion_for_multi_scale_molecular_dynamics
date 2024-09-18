import pytest
import torch

from crystal_diffusion.models.score_networks.force_field_augmented_score_network import (
    ForceFieldAugmentedScoreNetwork, ForceFieldParameters)
from crystal_diffusion.models.score_networks.mlp_score_network import (
    MLPScoreNetwork, MLPScoreNetworkParameters)
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)


@pytest.mark.parametrize("number_of_atoms", [4, 8, 16])
@pytest.mark.parametrize("radial_cutoff", [1.5, 2.0, 2.5])
class TestForceFieldAugmentedScoreNetwork:
    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(345345345)

    @pytest.fixture()
    def spatial_dimension(self):
        return 3

    @pytest.fixture()
    def score_network_parameters(self, number_of_atoms, spatial_dimension):
        # Generate an arbitrary MLP-based score network.
        return MLPScoreNetworkParameters(
            spatial_dimension=spatial_dimension,
            number_of_atoms=number_of_atoms,
            embedding_dimensions_size=12,
            n_hidden_dimensions=2,
            hidden_dimensions_size=16,
        )

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        return MLPScoreNetwork(score_network_parameters)

    @pytest.fixture()
    def force_field_parameters(self, radial_cutoff):
        return ForceFieldParameters(radial_cutoff=radial_cutoff, strength=1.0)

    @pytest.fixture()
    def force_field_augmented_score_network(
        self, score_network, force_field_parameters
    ):
        augmented_score_network = ForceFieldAugmentedScoreNetwork(
            score_network, force_field_parameters
        )
        return augmented_score_network

    @pytest.fixture()
    def batch_size(self):
        return 16

    @pytest.fixture
    def times(self, batch_size):
        times = torch.rand(batch_size, 1)
        return times

    @pytest.fixture()
    def basis_vectors(self, batch_size, spatial_dimension):
        # orthogonal boxes with dimensions between 5 and 10.
        orthogonal_boxes = torch.stack(
            [
                torch.diag(5.0 + 5.0 * torch.rand(spatial_dimension))
                for _ in range(batch_size)
            ]
        )
        # add a bit of noise to make the vectors not quite orthogonal
        basis_vectors = orthogonal_boxes + 0.1 * torch.randn(
            batch_size, spatial_dimension, spatial_dimension
        )
        return basis_vectors

    @pytest.fixture
    def relative_coordinates(
        self, batch_size, number_of_atoms, spatial_dimension, basis_vectors
    ):
        relative_coordinates = torch.rand(
            batch_size, number_of_atoms, spatial_dimension
        )
        return relative_coordinates

    @pytest.fixture
    def cartesian_forces(
        self, batch_size, number_of_atoms, spatial_dimension, basis_vectors
    ):
        cartesian_forces = torch.rand(batch_size, number_of_atoms, spatial_dimension)
        return cartesian_forces

    @pytest.fixture
    def noises(self, batch_size):
        return torch.rand(batch_size, 1)

    @pytest.fixture()
    def batch(
        self, relative_coordinates, cartesian_forces, times, noises, basis_vectors
    ):
        return {
            NOISY_RELATIVE_COORDINATES: relative_coordinates,
            TIME: times,
            UNIT_CELL: basis_vectors,
            NOISE: noises,
            CARTESIAN_FORCES: cartesian_forces,
        }

    @pytest.fixture
    def number_of_edges(self):
        return 128

    @pytest.fixture
    def fake_cartesian_displacements(self, number_of_edges, spatial_dimension):
        return torch.rand(number_of_edges, spatial_dimension)

    def test_get_cartesian_pseudo_forces_contributions(
        self,
        force_field_augmented_score_network,
        force_field_parameters,
        fake_cartesian_displacements,
    ):
        s = force_field_parameters.strength
        r0 = force_field_parameters.radial_cutoff

        expected_contributions = force_field_augmented_score_network._get_cartesian_pseudo_forces_contributions(
            fake_cartesian_displacements
        )

        for r, expected_contribution in zip(
            fake_cartesian_displacements, expected_contributions
        ):
            r_norm = torch.linalg.norm(r)

            r_hat = r / r_norm
            computed_contribution = -2.0 * s * (r_norm - r0) * r_hat
            torch.testing.assert_allclose(expected_contribution, computed_contribution)

    def test_get_cartesian_pseudo_forces(
        self, batch, force_field_augmented_score_network
    ):
        adj_info = force_field_augmented_score_network._get_adjacency_information(batch)
        cartesian_displacements = (
            force_field_augmented_score_network._get_cartesian_displacements(
                adj_info, batch
            )
        )
        cartesian_pseudo_force_contributions = (force_field_augmented_score_network.
                                                _get_cartesian_pseudo_forces_contributions(cartesian_displacements))

        computed_cartesian_pseudo_forces = (
            force_field_augmented_score_network._get_cartesian_pseudo_forces(
                cartesian_pseudo_force_contributions, adj_info, batch
            )
        )

        # Compute the expected value by explicitly looping over indices, effectively checking that
        # the 'torch.scatter_add' is used correctly.
        expected_cartesian_pseudo_forces = torch.zeros_like(
            computed_cartesian_pseudo_forces
        )
        batch_indices = adj_info.edge_batch_indices
        source_indices, _ = adj_info.adjacency_matrix
        for batch_idx, src_idx, cont in zip(
            batch_indices, source_indices, cartesian_pseudo_force_contributions
        ):
            expected_cartesian_pseudo_forces[batch_idx, src_idx] += cont

        torch.testing.assert_allclose(
            computed_cartesian_pseudo_forces, expected_cartesian_pseudo_forces
        )

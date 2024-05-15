import pytest
import torch
from e3nn import o3

from crystal_diffusion.models.mace_utils import \
    build_mace_output_nodes_irreducible_representation
from crystal_diffusion.models.score_prediction_head import (
    MaceEquivariantScorePredictionHead,
    MaceEquivariantScorePredictionHeadParameters)


class TestMaceEquivariantScorePredictionHead:

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(23423423)

    @pytest.fixture()
    def batch_size(self):
        return 16

    @pytest.fixture()
    def number_of_atoms(self):
        return 8

    @pytest.fixture()
    def num_interactions(self):
        return 2

    @pytest.fixture()
    def hidden_irreps_string(self):
        return "16x0e + 8x1o + 4x2e"

    @pytest.fixture()
    def output_node_features_irreps(self, hidden_irreps_string, num_interactions):
        output_node_features_irreps = (
            build_mace_output_nodes_irreducible_representation(hidden_irreps_string, num_interactions))
        return output_node_features_irreps

    @pytest.fixture()
    def parameters(self):
        return MaceEquivariantScorePredictionHeadParameters(time_embedding_irreps="4x0e", number_of_layers=2)

    @pytest.fixture()
    def prediction_head(self, output_node_features_irreps, parameters):
        return MaceEquivariantScorePredictionHead(output_node_features_irreps, parameters)

    @pytest.fixture()
    def times(self, batch_size):
        return torch.rand((batch_size,))

    @pytest.fixture()
    def flat_times(self, times, number_of_atoms):
        return torch.repeat_interleave(times, number_of_atoms).reshape(-1, 1)

    @pytest.fixture()
    def flat_node_features(self, batch_size, number_of_atoms, output_node_features_irreps):
        flat_batch_size = batch_size * number_of_atoms
        return output_node_features_irreps.randn(flat_batch_size, -1)

    def test_predictions_are_equivariant(self, prediction_head, flat_node_features,
                                         flat_times, output_node_features_irreps):

        vector_irreps = o3.Irreps("1x1o")
        random_rotation = o3.rand_matrix()

        d_in = output_node_features_irreps.D_from_matrix(random_rotation)
        d_out = vector_irreps.D_from_matrix(random_rotation)

        rotate_then_score = prediction_head(flat_node_features @ d_in.T, flat_times)
        score_then_rotate = prediction_head(flat_node_features, flat_times) @ d_out.T

        torch.testing.assert_close(rotate_then_score, score_then_rotate)

    def test_predictions_depend_on_time(self, prediction_head, flat_node_features):

        flat_batch_size = flat_node_features.shape[0]

        flat_time_1 = 0.25 * torch.ones(flat_batch_size, 1)
        flat_time_2 = 0.5 * torch.ones(flat_batch_size, 1)

        flat_score_1 = prediction_head(flat_node_features, flat_time_1)
        flat_score_2 = prediction_head(flat_node_features, flat_time_2)

        with pytest.raises(AssertionError):
            torch.testing.assert_close(flat_score_1, flat_score_2)

import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.egnn import E_GCL


class TestEGCL:

    @pytest.fixture(scope="class")
    def spatial_dimension(self):
        return 3

    @pytest.fixture(scope="class")
    def node_features_size(self):
        return 5

    @pytest.fixture(scope="class")
    def egcl_hyperparameters(self, node_features_size):
        hps = dict(
            input_size=node_features_size,
            message_n_hidden_dimensions=1,
            message_hidden_dimensions_size=4,
            node_n_hidden_dimensions=1,
            node_hidden_dimensions_size=4,
            coordinate_n_hidden_dimensions=1,
            coordinate_hidden_dimensions_size=4,
            output_size=node_features_size)
        return hps

    @pytest.fixture()
    def egcl(self, egcl_hyperparameters):
        model = E_GCL(**egcl_hyperparameters)
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def single_edge(self):
        return torch.Tensor([1, 0]).unsqueeze(0).long()

    @pytest.fixture(scope="class")
    def fixed_distance(self):
        return 0.4

    @pytest.fixture(scope="class")
    def simple_pair_coord(self, fixed_distance, spatial_dimension):
        coord = torch.zeros(2, spatial_dimension)
        coord[1, 0] = fixed_distance
        return coord

    def test_egcl_coord2radial(
        self, single_edge, fixed_distance, simple_pair_coord, egcl
    ):
        computed_distance_squared, computed_displacement = egcl.coord2radial(
            single_edge, simple_pair_coord
        )
        torch.testing.assert_close(computed_distance_squared.item(), fixed_distance**2)
        torch.testing.assert_close(
            computed_displacement, simple_pair_coord[1, :].unsqueeze(0)
        )

    def test_normalize_distance(self, egcl):

        zero = torch.tensor(0.)
        one = torch.tensor(1.)
        large = torch.tensor(1.0e3)

        torch.testing.assert_close(egcl.normalize_radial_norm(zero), zero)

        large_distance_norm = large * egcl.normalize_radial_norm(large**2)

        torch.testing.assert_close(large_distance_norm, one)

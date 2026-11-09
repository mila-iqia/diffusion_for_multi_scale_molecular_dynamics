import math

import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn import \
    E_GCL


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
            radial_cutoff=5.0,
            output_size=node_features_size)
        return hps

    @pytest.fixture()
    def egcl(self, egcl_hyperparameters):
        model = E_GCL(**egcl_hyperparameters)
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def single_edge(self):
        return torch.Tensor([1, 0, 0, 0, 0]).unsqueeze(0)

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

    def test_normalize_distance(self, egcl_hyperparameters):
        egcl = E_GCL(**egcl_hyperparameters, normalize="tanh")

        zero = torch.tensor(0.)
        one = torch.tensor(1.)
        large = torch.tensor(1.0e3)

        torch.testing.assert_close(egcl.normalize_radial_norm(zero), zero)

        large_distance_norm = large * egcl.normalize_radial_norm(large**2)

        torch.testing.assert_close(large_distance_norm, one)

    @pytest.mark.parametrize("distance", [1.0e-3, 0.1, 2.35])
    @pytest.mark.parametrize(
        "normalize, expected_norm",
        [
            ("raw", lambda r: r),
            ("tanh", lambda r: math.tanh(r**2)),
            ("smooth", lambda r: r / math.sqrt(r**2 + 0.1**2)),
        ],
    )
    def test_normalized_displacement(self, egcl_hyperparameters, single_edge, normalize, expected_norm, distance):
        """The normalized x_i - x_j keeps its direction and has the expected norm, with a = 0.1 Angstrom."""
        egcl = E_GCL(**egcl_hyperparameters, normalize=normalize, normalization_length=0.1)
        coord = torch.tensor([[0.0, 0.0, 0.0], [0.6 * distance, 0.8 * distance, 0.0]])
        _, displacement = egcl.coord2radial(single_edge, coord)
        norm = torch.linalg.norm(displacement)
        torch.testing.assert_close(norm.item(), expected_norm(distance), rtol=1e-4, atol=1e-9)
        torch.testing.assert_close(displacement / norm, coord[1:] / distance)

    def test_unknown_normalize_raises(self, egcl_hyperparameters):
        with pytest.raises(ValueError, match="normalize"):
            E_GCL(**egcl_hyperparameters, normalize="unit")

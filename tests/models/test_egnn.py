import math
from copy import copy

import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.egnn import (E_GCL,
                                                                      EGNN)


class TestEGNN:
    @pytest.fixture(scope="class", autouse=True)
    def set_default_type_to_float64(self):
        """Set the random seed."""
        torch.set_default_dtype(torch.float64)
        yield
        # this returns the default type to float32 at the end of all tests in this class in order
        # to not affect other tests.
        torch.set_default_dtype(torch.float32)

    @pytest.fixture(scope="class", autouse=True)
    def set_seed(self):
        """Set the random seed."""
        torch.manual_seed(234233)

    @pytest.fixture(scope="class")
    def batch_size(self):
        return 4

    @pytest.fixture(scope="class")
    def number_of_atoms(self):
        return 8

    @pytest.fixture(scope="class")
    def spatial_dimension(self):
        return 3

    @pytest.fixture(scope="class")
    def num_atom_types(self):
        return 5

    @pytest.fixture(scope="class")
    def relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension):
        relative_coordinates = torch.rand(
            batch_size, number_of_atoms, spatial_dimension
        )
        return relative_coordinates

    @pytest.fixture(scope="class")
    def node_features_size(self):
        return 5

    @pytest.fixture(scope="class")
    def node_features(self, batch_size, number_of_atoms, node_features_size):
        node_features = torch.randn(batch_size, number_of_atoms, node_features_size)
        return node_features

    @pytest.fixture(scope="class")
    def num_edges(self, number_of_atoms):
        return math.floor(number_of_atoms * 1.5)

    @pytest.fixture(scope="class")
    def edges(self, batch_size, number_of_atoms, num_edges):
        all_edges = []
        for b in range(batch_size):
            batch_edges = torch.Tensor(
                [(i, j) for i in range(number_of_atoms) for j in range(number_of_atoms)]
            )
            # select num_edges randomly
            indices = torch.randperm(len(batch_edges))
            shuffled_edges = batch_edges[indices] + b * number_of_atoms
            all_edges.append(shuffled_edges[:num_edges])
        return torch.cat(all_edges, dim=0).long()

    @pytest.fixture(scope="class")
    def batch(
        self, relative_coordinates, node_features, edges, batch_size, number_of_atoms
    ):
        batch = {
            "coord": relative_coordinates.view(batch_size * number_of_atoms, -1),
            "node_features": node_features.view(batch_size * number_of_atoms, -1),
            "edges": edges,
        }
        return batch

    @pytest.fixture(scope="class")
    def generic_hyperparameters(self, node_features_size):
        hps = dict(
            input_size=node_features_size,
            message_n_hidden_dimensions=1,
            message_hidden_dimensions_size=4,
            node_n_hidden_dimensions=1,
            node_hidden_dimensions_size=4,
            coordinate_n_hidden_dimensions=1,
            coordinate_hidden_dimensions_size=4,
        )
        return hps

    @pytest.fixture()
    def egnn_hyperparameters(self, generic_hyperparameters, num_atom_types):
        hps = copy(generic_hyperparameters)
        hps["n_layers"] = 2
        hps["num_classes"] = num_atom_types + 1
        return hps

    @pytest.fixture()
    def egcl_hyperparameters(self, generic_hyperparameters, node_features_size):
        hps = copy(generic_hyperparameters)
        hps["output_size"] = node_features_size
        return hps

    @pytest.fixture()
    def egcl(self, egcl_hyperparameters):
        model = E_GCL(**egcl_hyperparameters)
        model.eval()
        return model

    @pytest.fixture()
    def egnn(self, egnn_hyperparameters):
        model = EGNN(**egnn_hyperparameters)
        model.eval()
        return model

    @pytest.fixture()
    def egnn_scores(
        self,
        batch,
        egnn,
        batch_size,
        number_of_atoms,
        spatial_dimension,
        num_atom_types,
    ):
        egnn_scores = egnn(batch["node_features"], batch["edges"], batch["coord"])
        return {
            "X": egnn_scores.X.reshape(batch_size, number_of_atoms, spatial_dimension),
            "A": egnn_scores.A.reshape(batch_size, number_of_atoms, num_atom_types + 1),
        }

    @pytest.fixture()
    def egcl_scores(
        self,
        batch,
        egcl,
        batch_size,
        number_of_atoms,
        node_features_size,
        spatial_dimension,
    ):
        egcl_h, egcl_x = egcl(batch["node_features"], batch["edges"], batch["coord"])
        return egcl_h.reshape(
            batch_size, number_of_atoms, node_features_size
        ), egcl_x.reshape(batch_size, number_of_atoms, spatial_dimension)

    @pytest.fixture(scope="class")
    def permutations(self, batch_size, number_of_atoms):
        return torch.stack([torch.randperm(number_of_atoms) for _ in range(batch_size)])

    @pytest.fixture(scope="class")
    def permuted_coordinates(self, batch_size, number_of_atoms, batch, permutations):
        permuted_batch = batch
        pos = permuted_batch["coord"].view(batch_size, number_of_atoms, -1)
        permuted_pos = torch.stack(
            [
                pos[batch_idx, permutations[batch_idx], :]
                for batch_idx in range(batch_size)
            ]
        )
        return permuted_pos.view(batch_size * number_of_atoms, -1)

    @pytest.fixture(scope="class")
    def permuted_node_features(self, batch_size, number_of_atoms, batch, permutations):
        permuted_batch = batch

        h = permuted_batch["node_features"].view(batch_size, number_of_atoms, -1)
        permuted_h = torch.stack(
            [
                h[batch_idx, permutations[batch_idx], :]
                for batch_idx in range(batch_size)
            ]
        )
        return permuted_h.view(batch_size * number_of_atoms, -1)

    @pytest.fixture(scope="class")
    def permuted_edges(self, batch_size, batch, permutations, number_of_atoms):
        edges = batch["edges"]
        permuted_edges = edges.clone()
        for b in range(batch_size):
            for atom in range(number_of_atoms):
                new_atom_idx = permutations[b, atom] + b * number_of_atoms
                permuted_edges[edges == new_atom_idx] = atom + b * number_of_atoms
        return permuted_edges.long()

    @pytest.fixture()
    def permuted_batch(
        self, permuted_coordinates, permuted_edges, permuted_node_features
    ):
        permuted_batch = {
            "coord": permuted_coordinates,
            "node_features": permuted_node_features,
            "edges": permuted_edges,
        }
        return permuted_batch

    @pytest.fixture()
    def permuted_egnn_scores(
        self,
        permuted_batch,
        egnn,
        batch_size,
        number_of_atoms,
        spatial_dimension,
        num_atom_types,
    ):
        egnn_scores = egnn(
            permuted_batch["node_features"],
            permuted_batch["edges"],
            permuted_batch["coord"],
        )
        return {
            "X": egnn_scores.X.reshape(batch_size, number_of_atoms, spatial_dimension),
            "A": egnn_scores.A.reshape(batch_size, number_of_atoms, num_atom_types + 1),
        }

    @pytest.fixture()
    def permuted_egcl_scores(self, permuted_batch, egcl, batch_size, number_of_atoms):
        egcl_h, egcl_x = egcl(
            permuted_batch["node_features"],
            permuted_batch["edges"],
            permuted_batch["coord"],
        )
        return egcl_h.reshape(batch_size, number_of_atoms, -1), egcl_x.reshape(
            batch_size, number_of_atoms, -1
        )

    def test_egcl_permutation_equivariance(
        self, egcl_scores, permuted_egcl_scores, batch_size, permutations
    ):
        permuted_egcl_h, permuted_egcl_x = permuted_egcl_scores
        egcl_h, egcl_x = egcl_scores

        expected_permuted_h = torch.stack(
            [
                egcl_h[batch_idx, permutations[batch_idx], :]
                for batch_idx in range(batch_size)
            ]
        )

        torch.testing.assert_close(expected_permuted_h, permuted_egcl_h)

        expected_permuted_x = torch.stack(
            [
                egcl_x[batch_idx, permutations[batch_idx], :]
                for batch_idx in range(batch_size)
            ]
        )

        torch.testing.assert_close(expected_permuted_x, permuted_egcl_x)

    def test_egnn_permutation_equivariance(
        self, egnn_scores, permuted_egnn_scores, batch_size, permutations
    ):
        expected_permuted_scores = {
            "X": torch.stack(
                [
                    egnn_scores["X"][batch_idx, permutations[batch_idx], :]
                    for batch_idx in range(batch_size)
                ]
            ),
            "A": torch.stack(
                [
                    egnn_scores["A"][batch_idx, permutations[batch_idx], :]
                    for batch_idx in range(batch_size)
                ]
            ),
        }

        torch.testing.assert_close(
            expected_permuted_scores["X"], permuted_egnn_scores["X"]
        )
        torch.testing.assert_close(
            expected_permuted_scores["A"], permuted_egnn_scores["A"]
        )

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

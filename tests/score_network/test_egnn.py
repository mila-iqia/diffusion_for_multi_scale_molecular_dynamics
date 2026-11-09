import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISE, NOISY_AXL_COMPOSITION)
from diffusion_for_multi_scale_molecular_dynamics.score_network.cartesian_egnn import (
    CartesianE_GCL, CartesianEGNN)
from diffusion_for_multi_scale_molecular_dynamics.score_network.cartesian_egnn_score_network import (
    CartesianEGNNScoreNetwork, CartesianEGNNScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn import \
    E_GCL
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn_score_network import \
    EGNNScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn_utils import (
    unsorted_segment_mean, unsorted_segment_sum)
from diffusion_for_multi_scale_molecular_dynamics.score_network.torus_egnn import \
    TorusEGNN


@pytest.fixture(scope="module", autouse=True)
def set_random_seed():
    torch.manual_seed(0)


@pytest.fixture()
def radial_cutoff():
    return 3.0


@pytest.fixture()
def node_features_size():
    return 5


@pytest.fixture()
def egcl_hyperparameters(node_features_size, radial_cutoff):
    return dict(
        input_size=node_features_size,
        output_size=node_features_size,
        message_n_hidden_dimensions=1,
        message_hidden_dimensions_size=8,
        node_n_hidden_dimensions=1,
        node_hidden_dimensions_size=8,
        coordinate_n_hidden_dimensions=1,
        coordinate_hidden_dimensions_size=8,
        radial_cutoff=radial_cutoff,
    )


@pytest.fixture()
def egnn_hyperparameters(radial_cutoff):
    return dict(
        input_size=3,
        num_classes=2,
        message_n_hidden_dimensions=1,
        message_hidden_dimensions_size=8,
        node_n_hidden_dimensions=1,
        node_hidden_dimensions_size=8,
        coordinate_n_hidden_dimensions=1,
        coordinate_hidden_dimensions_size=8,
        radial_cutoff=radial_cutoff,
    )


class TestCutoffEnvelope:

    @pytest.fixture()
    def egcl(self, egcl_hyperparameters):
        return E_GCL(**egcl_hyperparameters)

    def test_envelope_values(self, egcl, radial_cutoff):
        distance = torch.tensor([0.0, 0.5 * radial_cutoff, radial_cutoff, 1.5 * radial_cutoff])
        envelope = egcl.cutoff_envelope((distance**2).unsqueeze(1))
        expected = torch.tensor([1.0, 0.5, 0.0, 0.0]).unsqueeze(1)
        torch.testing.assert_close(envelope, expected)

    def test_envelope_decreases_inside_cutoff(self, egcl, radial_cutoff):
        distance = torch.linspace(0.01, 0.99 * radial_cutoff, 100)
        envelope = egcl.cutoff_envelope((distance**2).unsqueeze(1)).squeeze(1)
        assert torch.all(envelope[1:] < envelope[:-1])
        assert torch.all(envelope > 0.0)


class TestAggregate:

    @pytest.fixture()
    def egcl(self, egcl_hyperparameters):
        return E_GCL(**egcl_hyperparameters)

    @pytest.fixture()
    def num_nodes(self):
        return 4

    @pytest.fixture()
    def row(self):
        return torch.tensor([0, 0, 0, 1, 1, 2])

    @pytest.fixture()
    def values(self, row):
        return torch.randn(len(row), 3)

    @pytest.fixture()
    def envelope(self):
        return torch.tensor([1.0, 0.5, 0.25, 0.2, 0.0, 0.0]).unsqueeze(1)

    def test_weighted_mean(self, egcl, values, row, num_nodes, envelope):
        computed = egcl._aggregate(values, row, num_nodes, "mean", envelope)
        expected = torch.zeros(num_nodes, values.shape[1])
        for node in range(num_nodes):
            mask = row == node
            total_weight = envelope[mask].sum()
            if total_weight > 0:
                expected[node] = (envelope[mask] * values[mask]).sum(dim=0) / total_weight
        torch.testing.assert_close(computed, expected)

    def test_weighted_sum(self, egcl, values, row, num_nodes, envelope):
        computed = egcl._aggregate(values, row, num_nodes, "sum", envelope)
        expected = torch.zeros(num_nodes, values.shape[1])
        for node in range(num_nodes):
            mask = row == node
            expected[node] = (envelope[mask] * values[mask]).sum(dim=0)
        torch.testing.assert_close(computed, expected)

    @pytest.mark.parametrize("aggregation", ["mean", "sum"])
    def test_zero_weight_nodes_are_zero(self, egcl, values, row, num_nodes, envelope, aggregation):
        computed = egcl._aggregate(values, row, num_nodes, aggregation, envelope)
        torch.testing.assert_close(computed[2:], torch.zeros(2, values.shape[1]))

    @pytest.mark.parametrize(
        "aggregation, aggregation_fn", [("mean", unsorted_segment_mean), ("sum", unsorted_segment_sum)]
    )
    def test_no_envelope_is_plain_aggregation(self, egcl, values, row, num_nodes, aggregation, aggregation_fn):
        computed = egcl._aggregate(values, row, num_nodes, aggregation, None)
        expected = aggregation_fn(values, row, num_segments=num_nodes)
        torch.testing.assert_close(computed, expected)


class TestEGCLWithoutSmoothCutoff:

    @pytest.fixture()
    def egcl(self, egcl_hyperparameters):
        model = E_GCL(**egcl_hyperparameters, smooth_cutoff=False)
        model.eval()
        return model

    @pytest.fixture()
    def num_nodes(self):
        return 4

    @pytest.fixture()
    def h(self, num_nodes, node_features_size):
        return torch.randn(num_nodes, node_features_size)

    @pytest.fixture()
    def coord(self, num_nodes):
        return torch.randn(num_nodes, 3)

    @pytest.fixture()
    def edge_index(self):
        pairs = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1]]).float()
        shifts = torch.zeros(len(pairs), 3)
        shifts[4:, 0] = torch.tensor([1.0, -1.0])
        return torch.cat([pairs, shifts], dim=1)

    def test_forward_matches_plain_mean_aggregation(self, egcl, h, coord, edge_index, num_nodes):
        row, col = edge_index[:, 0].long(), edge_index[:, 1].long()
        coord_diff = coord[row] - coord[col] - edge_index[:, 2:]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        with torch.no_grad():
            messages = egcl.message_model(h[row], h[col], radial)
            expected_coord = coord + unsorted_segment_mean(
                coord_diff * egcl.coord_mlp(messages), row, num_segments=num_nodes
            )
            aggregated_messages = unsorted_segment_mean(messages, row, num_segments=num_nodes)
            expected_h = h + egcl.node_mlp(torch.cat([h, aggregated_messages], dim=1))

            computed_h, computed_coord = egcl(h, edge_index, coord.clone(), radial)

        torch.testing.assert_close(computed_h, expected_h)
        torch.testing.assert_close(computed_coord, expected_coord)

    def test_edges_without_shift(self, egcl, coord, edge_index):
        """Edges given as [i, j] only: x_i - x_j, with no periodic shift."""
        _, coord_diff = egcl.coord2radial(edge_index[:, :2], coord)
        row, col = edge_index[:, 0].long(), edge_index[:, 1].long()
        torch.testing.assert_close(coord_diff, coord[row] - coord[col])


class TestEGCLEnvelopeUsesGivenDistances:
    """The smooth cutoff uses the edge distances given to the layer, not the distances between the node positions."""

    @pytest.fixture()
    def egcl(self, egcl_hyperparameters):
        model = E_GCL(**egcl_hyperparameters, smooth_cutoff=True)
        model.eval()
        return model

    @pytest.fixture()
    def h(self, node_features_size):
        return torch.randn(2, node_features_size)

    @pytest.fixture()
    def coord(self):
        return torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    @pytest.fixture()
    def edge_index(self):
        return torch.tensor([[0, 1], [1, 0]]).float()

    def test_edges_beyond_the_cutoff_do_not_contribute(self, egcl, h, coord, edge_index, radial_cutoff):
        """The nodes are 1 A apart, but the given distance is beyond the cutoff: no message, no displacement."""
        beyond_cutoff = torch.full((2, 1), (1.1 * radial_cutoff) ** 2)
        with torch.no_grad():
            computed_h, computed_coord = egcl(h, edge_index, coord.clone(), beyond_cutoff)
            expected_h = h + egcl.node_mlp(torch.cat([h, torch.zeros(2, egcl.message_mlp[-2].out_features)], dim=1))
        torch.testing.assert_close(computed_coord, coord)
        torch.testing.assert_close(computed_h, expected_h)

    def test_envelope_weights_the_messages(self, egcl, h, coord, edge_index, radial_cutoff):
        """With a single neighbour, the weighted mean does not depend on the given distance inside the cutoff."""
        with torch.no_grad():
            close = egcl(h, edge_index, coord.clone(), torch.full((2, 1), (0.2 * radial_cutoff) ** 2))
            far = egcl(h, edge_index, coord.clone(), torch.full((2, 1), (0.8 * radial_cutoff) ** 2))
        torch.testing.assert_close(close[0], far[0])
        torch.testing.assert_close(close[1], far[1])


class TestEGNNContinuityAtCutoff:

    @pytest.fixture()
    def step(self):
        return 1.0e-3

    @pytest.fixture()
    def distances(self, radial_cutoff, step):
        return torch.arange(radial_cutoff - 0.2, radial_cutoff + 0.2, step)

    @pytest.fixture()
    def cell_lengths(self):
        return torch.tensor([[20.0, 20.0, 20.0]])

    @pytest.fixture()
    def h(self):
        return torch.randn(4, 3)

    def compute_outputs(self, egnn, h, cell_lengths, distances):
        outputs = []
        with torch.no_grad():
            for distance in distances:
                x = torch.tensor(
                    [
                        [5.0, 5.0, 5.0],
                        [5.0 + distance, 5.0, 5.0],
                        [5.0, 6.5, 5.0],
                        [6.5 + distance, 5.0, 5.0],
                    ]
                )
                model_outputs = egnn(h, x, cell_lengths)
                outputs.append(torch.cat([model_outputs.X.flatten(), model_outputs.A.flatten()]))
        outputs = torch.stack(outputs)
        return (outputs[1:] - outputs[:-1]).abs().max(dim=1).values

    @pytest.mark.parametrize("n_layers", [1, 2])
    @pytest.mark.parametrize("aggregation", ["mean", "sum"])
    def test_smooth_cutoff_is_continuous(
        self, egnn_hyperparameters, h, cell_lengths, distances, step, n_layers, aggregation
    ):
        egnn = CartesianEGNN(
            **egnn_hyperparameters, n_layers=n_layers, coords_agg=aggregation, message_agg=aggregation,
            smooth_cutoff=True,
        )
        egnn.eval()
        output_changes = self.compute_outputs(egnn, h, cell_lengths, distances)
        assert output_changes.max() < 5.0 * step

    def test_hard_cutoff_is_discontinuous(self, egnn_hyperparameters, h, cell_lengths, distances):
        egnn = CartesianEGNN(**egnn_hyperparameters, n_layers=1, smooth_cutoff=False)
        egnn.eval()
        output_changes = self.compute_outputs(egnn, h, cell_lengths, distances)
        assert output_changes.max() > 2.0e-2


class TestRebuildEdgesEveryLayer:

    @pytest.fixture()
    def batch_size(self):
        return 2

    @pytest.fixture()
    def number_of_atoms(self):
        return 6

    @pytest.fixture()
    def cell_lengths(self, batch_size):
        return torch.tensor([[5.0, 5.0, 5.0], [6.0, 5.5, 5.0]])[:batch_size]

    @pytest.fixture()
    def h(self, batch_size, number_of_atoms):
        return torch.randn(batch_size * number_of_atoms, 3)

    @pytest.fixture()
    def x(self, batch_size, number_of_atoms, cell_lengths):
        relative_coordinates = torch.rand(batch_size, number_of_atoms, 3)
        return (relative_coordinates * cell_lengths.unsqueeze(1)).reshape(-1, 3)

    @pytest.mark.parametrize("rebuild_edges_every_layer, expected_calls", [(True, 3), (False, 1)])
    def test_number_of_edge_builds(
        self, egnn_hyperparameters, h, x, cell_lengths, rebuild_edges_every_layer, expected_calls
    ):
        egnn = CartesianEGNN(**egnn_hyperparameters, n_layers=3, rebuild_edges_every_layer=rebuild_edges_every_layer)
        original_build_edges = egnn._build_edges
        calls = []

        def counting_build_edges(*args, **kwargs):
            calls.append(1)
            return original_build_edges(*args, **kwargs)

        egnn._build_edges = counting_build_edges
        egnn(h, x, cell_lengths)
        assert len(calls) == expected_calls

    def test_single_layer_is_independent_of_rebuild(self, egnn_hyperparameters, h, x, cell_lengths):
        egnn_rebuild = CartesianEGNN(**egnn_hyperparameters, n_layers=1, rebuild_edges_every_layer=True)
        egnn_once = CartesianEGNN(**egnn_hyperparameters, n_layers=1, rebuild_edges_every_layer=False)
        egnn_once.load_state_dict(egnn_rebuild.state_dict())
        egnn_rebuild.eval()
        egnn_once.eval()
        with torch.no_grad():
            outputs_rebuild = egnn_rebuild(h, x.clone(), cell_lengths)
            outputs_once = egnn_once(h, x.clone(), cell_lengths)
        torch.testing.assert_close(outputs_rebuild.X, outputs_once.X)
        torch.testing.assert_close(outputs_rebuild.A, outputs_once.A)


class TestTorusEGNNMatchesCartesianEGNN:
    """With the same weights and the same fixed edges, both EGNN subclasses apply the same layers.

    The cell is large enough that no periodic image is within the cutoff, so the cartesian edges have no shift. With
    a smooth cutoff, only one layer is compared: CartesianEGNN updates the edge distances from the layer-updated
    positions, while TorusEGNN keeps the cartesian distances of its input. The bond direction feature is off, since
    the torus layers only see the distance squared.
    """

    @pytest.fixture()
    def number_of_atoms(self):
        return 6

    @pytest.fixture()
    def cell_lengths(self):
        return torch.tensor([[30.0, 30.0, 30.0]])

    @pytest.fixture()
    def h(self, number_of_atoms):
        return torch.randn(number_of_atoms, 3)

    @pytest.fixture()
    def x(self, number_of_atoms):
        return 10.0 + 2.0 * torch.rand(number_of_atoms, 3)

    @pytest.mark.parametrize("smooth_cutoff, n_layers", [(False, 3), (True, 1)])
    def test_same_outputs(self, egnn_hyperparameters, h, x, cell_lengths, smooth_cutoff, n_layers):
        cartesian_egnn = CartesianEGNN(
            **egnn_hyperparameters, n_layers=n_layers, smooth_cutoff=smooth_cutoff, rebuild_edges_every_layer=False,
            bond_direction_feature=False,
        )
        torus_egnn = TorusEGNN(**egnn_hyperparameters, n_layers=n_layers, smooth_cutoff=smooth_cutoff)
        torus_egnn.load_state_dict(cartesian_egnn.state_dict())
        cartesian_egnn.eval()
        torus_egnn.eval()

        edges = cartesian_egnn._build_edges(x, cell_lengths, natoms=None)
        assert torch.all(edges[:, 2:] == 0.0)
        edge_distances_squared = CartesianEGNN._get_edge_distances_squared(edges, x)
        with torch.no_grad():
            cartesian_outputs = cartesian_egnn(h, x.clone(), cell_lengths)
            torus_outputs = torus_egnn(h, x.clone(), edges[:, :2], edge_distances_squared)
        torch.testing.assert_close(torus_outputs.X, cartesian_outputs.X)
        torch.testing.assert_close(torus_outputs.A, cartesian_outputs.A)


class TestBondDirectionFeature:
    """The cartesian layers also give the messages the bond direction (dx^4 + dy^4 + dz^4) / (d^2 + a^2)^2."""

    @pytest.fixture()
    def egcl(self, egcl_hyperparameters):
        return CartesianE_GCL(**egcl_hyperparameters)

    @staticmethod
    def get_features(egcl, coord_diff):
        radial = (coord_diff**2).sum(dim=1, keepdim=True)
        return egcl.get_edge_features(radial, coord_diff)

    @pytest.mark.parametrize(
        "direction, expected_direction_invariant",
        [([1.0, 0.0, 0.0], 1.0), ([0.0, -1.0, 0.0], 1.0), ([1.0, 1.0, 1.0], 1.0 / 3.0), ([1.0, -1.0, 0.0], 0.5)],
    )
    def test_feature_values(self, egcl, direction, expected_direction_invariant):
        distance = 2.35
        coord_diff = distance * torch.nn.functional.normalize(torch.tensor([direction]), dim=-1)
        features = self.get_features(egcl, coord_diff)
        smoothing = (distance**2 / (distance**2 + CartesianE_GCL.bond_direction_length**2)) ** 2
        torch.testing.assert_close(features[:, 0], torch.tensor([distance**2]))
        torch.testing.assert_close(features[:, 1], torch.tensor([expected_direction_invariant * smoothing]))

    def test_overlapping_atoms_have_a_vanishing_feature(self, egcl):
        directions = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
        for distance in [0.0, 1.0e-4, 1.0e-3]:
            features = self.get_features(egcl, distance * directions)
            assert torch.all(torch.isfinite(features))
            assert torch.all(features[:, 1] <= distance**4 / CartesianE_GCL.bond_direction_length**4 + 1e-12)

    def test_feature_is_invariant_under_the_cubic_symmetries_only(self, egcl):
        coord_diff = torch.randn(20, 3)
        features = self.get_features(egcl, coord_diff)
        for permutation in [[1, 0, 2], [2, 0, 1]]:
            for signs in [[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]]:
                transformed = coord_diff[:, permutation] * torch.tensor(signs)
                torch.testing.assert_close(self.get_features(egcl, transformed), features)
        angle = torch.tensor(0.3)
        rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0],
                                 [0.0, 0.0, 1.0]])
        rotated_features = self.get_features(egcl, coord_diff @ rotation.T)
        torch.testing.assert_close(rotated_features[:, 0], features[:, 0])
        assert not torch.allclose(rotated_features[:, 1], features[:, 1])

    @pytest.mark.parametrize("normalize", ["raw", "tanh", "smooth"])
    def test_feature_uses_the_raw_difference(self, mocker, egcl_hyperparameters, node_features_size, normalize):
        egcl = CartesianE_GCL(**egcl_hyperparameters, normalize=normalize)
        message_model_spy = mocker.spy(egcl, "message_model")
        coord = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.5]])
        edge_index = torch.tensor([[0, 1], [1, 0]]).float()
        raw_coord_diff = torch.stack([coord[0] - coord[1], coord[1] - coord[0]])
        radial = (raw_coord_diff**2).sum(dim=1, keepdim=True)
        with torch.no_grad():
            egcl(torch.randn(2, node_features_size), edge_index, coord.clone(), radial)
        edge_features = message_model_spy.call_args.args[2]
        torch.testing.assert_close(edge_features, egcl.get_edge_features(radial, raw_coord_diff))

    @pytest.mark.parametrize("layer_class, number_of_edge_features", [(E_GCL, 1), (CartesianE_GCL, 2)])
    def test_message_input_size(self, egcl_hyperparameters, node_features_size, layer_class, number_of_edge_features):
        egcl = layer_class(**egcl_hyperparameters)
        assert egcl.message_mlp[0].in_features == 2 * node_features_size + number_of_edge_features

    @pytest.mark.parametrize("bond_direction_feature, layer_class", [(True, CartesianE_GCL), (False, E_GCL)])
    def test_option_reaches_the_layers(self, bond_direction_feature, layer_class):
        score_network_parameters = CartesianEGNNScoreNetworkParameters(
            radial_cutoff=3.0, num_atom_types=1, bond_direction_feature=bond_direction_feature,
        )
        score_network = CartesianEGNNScoreNetwork(score_network_parameters)
        for graph_layer in score_network.egnn.graph_layers:
            assert type(graph_layer) is layer_class

    def test_on_by_default(self):
        assert CartesianEGNNScoreNetworkParameters(radial_cutoff=3.0, num_atom_types=1).bond_direction_feature


@pytest.mark.parametrize("smooth_cutoff", [True, False])
@pytest.mark.parametrize("rebuild_edges_every_layer", [True, False])
def test_score_network_passes_options_to_egnn(smooth_cutoff, rebuild_edges_every_layer):
    score_network_parameters = CartesianEGNNScoreNetworkParameters(
        radial_cutoff=3.0,
        num_atom_types=1,
        smooth_cutoff=smooth_cutoff,
        rebuild_edges_every_layer=rebuild_edges_every_layer,
    )
    score_network = CartesianEGNNScoreNetwork(score_network_parameters)
    assert isinstance(score_network.egnn, CartesianEGNN)
    assert score_network.egnn.rebuild_edges_every_layer == rebuild_edges_every_layer
    for graph_layer in score_network.egnn.graph_layers:
        assert graph_layer.smooth_cutoff == smooth_cutoff
        assert graph_layer.radial_cutoff == 3.0


@pytest.mark.parametrize("normalize", ["raw", "tanh", "smooth"])
def test_score_network_passes_normalize_to_egnn(normalize):
    score_network_parameters = CartesianEGNNScoreNetworkParameters(
        radial_cutoff=3.0, num_atom_types=1, normalize=normalize, normalization_length=0.2,
    )
    score_network = CartesianEGNNScoreNetwork(score_network_parameters)
    for graph_layer in score_network.egnn.graph_layers:
        assert graph_layer.normalize == normalize
        assert graph_layer.normalization_length == 0.2


def test_score_network_options_default_to_true():
    score_network_parameters = CartesianEGNNScoreNetworkParameters(radial_cutoff=3.0, num_atom_types=1)
    assert score_network_parameters.architecture == "egnn_cartesian"
    assert score_network_parameters.smooth_cutoff
    assert score_network_parameters.rebuild_edges_every_layer


@pytest.mark.parametrize("noise_input", ["log", "raw"])
def test_node_attributes_noise_input(noise_input):
    sigmas = torch.tensor([[0.01], [2.0]])
    number_of_atoms = 3
    batch = {
        NOISY_AXL_COMPOSITION: AXL(A=torch.zeros(2, number_of_atoms, dtype=torch.long),
                                   X=torch.rand(2, number_of_atoms, 3), L=torch.ones(2, 6)),
        NOISE: sigmas,
    }
    node_attributes = EGNNScoreNetwork._get_node_attributes(batch, num_atom_types=1, noise_input=noise_input)

    expected_noise_feature = torch.log(sigmas) if noise_input == "log" else sigmas
    expected_noise_feature_per_atom = expected_noise_feature.repeat_interleave(number_of_atoms, dim=0)
    torch.testing.assert_close(node_attributes[:, :1], expected_noise_feature_per_atom)

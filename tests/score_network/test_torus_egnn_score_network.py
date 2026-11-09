import itertools

import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISE, NOISY_AXL_COMPOSITION, NUMBER_OF_ATOMS, TIME)
from diffusion_for_multi_scale_molecular_dynamics.score_network.torus_egnn import \
    TorusEGNN
from diffusion_for_multi_scale_molecular_dynamics.score_network.torus_egnn_score_network import (
    TorusEGNNScoreNetwork, TorusEGNNScoreNetworkParameters)


@pytest.fixture(scope="module", autouse=True)
def set_random_seed():
    torch.manual_seed(0)


def make_batch(relative_coordinates, cell_lengths, natoms=None):
    """Batch of single-type atoms in orthogonal cells, for relative coordinates [batch_size, atoms, 3]."""
    batch_size, number_of_atoms, _ = relative_coordinates.shape
    lattice_parameters = torch.cat([cell_lengths, torch.zeros(batch_size, 3)], dim=1)
    return {
        NOISY_AXL_COMPOSITION: AXL(A=torch.zeros(batch_size, number_of_atoms, dtype=torch.long),
                                   X=relative_coordinates, L=lattice_parameters),
        NOISE: torch.full((batch_size, 1), 0.1),
        TIME: torch.full((batch_size, 1), 0.5),
        NUMBER_OF_ATOMS: torch.full((batch_size,), number_of_atoms) if natoms is None else natoms,
    }


def make_score_network(**options):
    parameters = TorusEGNNScoreNetworkParameters(
        num_atom_types=1, n_layers=2, message_hidden_dimensions_size=8, node_hidden_dimensions_size=8,
        coordinate_hidden_dimensions_size=8, **options,
    )
    return TorusEGNNScoreNetwork(parameters).eval()


def minimum_image_distances_squared(cartesian_positions, cell_lengths):
    """Minimum image distance squared between every pair of atoms of one orthogonal cell."""
    displacements = cartesian_positions[:, None, :] - cartesian_positions[None, :, :]
    displacements = displacements - cell_lengths * torch.round(displacements / cell_lengths)
    return (displacements**2).sum(dim=-1)


def expected_edges(cartesian_positions, cell_lengths, natoms, radial_cutoff=None):
    """Every pair of distinct real atoms (within the radial cutoff, if given), with its minimum image distance."""
    batch_size, max_atoms, _ = cartesian_positions.shape
    edges = {}
    for sample in range(batch_size):
        distances_squared = minimum_image_distances_squared(cartesian_positions[sample], cell_lengths[sample])
        for i, j in itertools.permutations(range(int(natoms[sample])), 2):
            if radial_cutoff is None or distances_squared[i, j] <= radial_cutoff**2:
                edges[(i + sample * max_atoms, j + sample * max_atoms)] = float(distances_squared[i, j])
    return edges


def edges_to_dictionary(edges, edge_distances_squared):
    return {(int(i), int(j)): float(d) for (i, j), d in zip(edges.long().tolist(), edge_distances_squared[:, 0])}


class TestEdges:

    @pytest.fixture(params=[5.44, 10.89])
    def cell_lengths(self, request):
        return torch.tensor([[request.param] * 3, [request.param * 1.1, request.param, request.param * 0.9]])

    @pytest.fixture()
    def max_atoms(self):
        return 8

    @pytest.fixture()
    def natoms(self, max_atoms):
        return torch.tensor([max_atoms, max_atoms - 2])

    @pytest.fixture()
    def cartesian_positions(self, cell_lengths, max_atoms, natoms):
        positions = torch.rand(len(cell_lengths), max_atoms, 3) * cell_lengths[:, None, :]
        positions[1, natoms[1]:] = float("nan")
        return positions

    def test_closest_image_edges(self, cartesian_positions, cell_lengths, natoms):
        """One edge per pair of distinct real atoms within the cutoff, at its minimum image distance.

        In the 5.44 A cell, the 5 A cutoff is larger than half the cell: some pairs have several images within the
        cutoff, but appear once.
        """
        score_network = make_score_network(radial_cutoff=5.0)
        edges, edge_distances_squared = score_network._get_closest_image_edges(
            cartesian_positions, cell_lengths, natoms
        )
        computed = edges_to_dictionary(edges, edge_distances_squared)
        expected = expected_edges(cartesian_positions, cell_lengths, natoms, radial_cutoff=5.0)
        assert len(computed) == len(edges)
        assert computed.keys() == expected.keys()
        for pair, distance_squared in expected.items():
            assert computed[pair] == pytest.approx(distance_squared, rel=1e-5, abs=1e-6)

    def test_fully_connected_edges(self, cartesian_positions, cell_lengths, natoms):
        """Every pair of distinct real atoms, once, at its minimum image distance."""
        edges, edge_distances_squared = TorusEGNNScoreNetwork._get_fully_connected_edges(
            cartesian_positions, cell_lengths, natoms
        )
        computed = edges_to_dictionary(edges, edge_distances_squared)
        expected = expected_edges(cartesian_positions, cell_lengths, natoms)
        assert len(computed) == len(edges)
        assert computed.keys() == expected.keys()
        for pair, distance_squared in expected.items():
            assert computed[pair] == pytest.approx(distance_squared, rel=1e-5, abs=1e-6)


class TestScoreNetwork:

    @pytest.fixture()
    def number_of_atoms(self):
        return 8

    @pytest.fixture()
    def cell_lengths(self):
        return torch.tensor([[5.44, 5.44, 5.44], [5.6, 5.3, 5.5]])

    @pytest.fixture()
    def relative_coordinates(self, number_of_atoms):
        return torch.rand(2, number_of_atoms, 3)

    @pytest.mark.parametrize(
        "edges, radial_cutoff, smooth_cutoff",
        [("fully_connected", None, False), ("radial_cutoff", 5.0, True), ("radial_cutoff", 5.0, False)],
    )
    def test_options_reach_the_egnn(self, edges, radial_cutoff, smooth_cutoff):
        score_network = make_score_network(edges=edges, radial_cutoff=radial_cutoff, smooth_cutoff=smooth_cutoff)
        assert score_network.edges == edges
        assert isinstance(score_network.egnn, TorusEGNN)
        for graph_layer in score_network.egnn.graph_layers:
            assert graph_layer.smooth_cutoff == smooth_cutoff
            assert graph_layer.radial_cutoff == radial_cutoff

    def test_default_parameters(self):
        parameters = TorusEGNNScoreNetworkParameters(num_atom_types=1, radial_cutoff=5.0)
        assert parameters.architecture == "egnn_torus"
        assert parameters.edges == "radial_cutoff"
        assert parameters.smooth_cutoff

    def test_fully_connected_matches_a_cutoff_beyond_every_distance(self, relative_coordinates, cell_lengths):
        """Without the smooth cutoff, a radial cutoff larger than every minimum image distance keeps every pair."""
        fully_connected = make_score_network(edges="fully_connected", radial_cutoff=None, smooth_cutoff=False)
        radial_cutoff = make_score_network(edges="radial_cutoff", radial_cutoff=10.0, smooth_cutoff=False)
        radial_cutoff.load_state_dict(fully_connected.state_dict())
        with torch.no_grad():
            fully_connected_scores = fully_connected(make_batch(relative_coordinates, cell_lengths))
            radial_cutoff_scores = radial_cutoff(make_batch(relative_coordinates, cell_lengths))
        torch.testing.assert_close(fully_connected_scores.X, radial_cutoff_scores.X)
        torch.testing.assert_close(fully_connected_scores.A, radial_cutoff_scores.A)

    @pytest.mark.parametrize(
        "edges, radial_cutoff, smooth_cutoff", [("fully_connected", None, False), ("radial_cutoff", 5.0, True)]
    )
    def test_moving_atoms_by_lattice_vectors(self, relative_coordinates, cell_lengths, edges, radial_cutoff,
                                             smooth_cutoff):
        """The torus embedding and the minimum image distances do not change when atoms move by lattice vectors."""
        score_network = make_score_network(edges=edges, radial_cutoff=radial_cutoff, smooth_cutoff=smooth_cutoff)
        lattice_vectors = torch.randint(-2, 3, relative_coordinates.shape).float()
        with torch.no_grad():
            scores = score_network._forward_unchecked(make_batch(relative_coordinates, cell_lengths))
            moved_scores = score_network._forward_unchecked(
                make_batch(relative_coordinates + lattice_vectors, cell_lengths)
            )
        torch.testing.assert_close(scores.X, moved_scores.X, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(scores.A, moved_scores.A, atol=1e-5, rtol=1e-4)


class TestSmoothCutoffOnCartesianDistances:
    """An atom crossing the radial cutoff changes the scores continuously only with the smooth cutoff.

    The positions given to EGNN are on the torus, so the envelope must use the cartesian distance of every edge.
    """

    @pytest.fixture()
    def radial_cutoff(self):
        return 3.0

    @pytest.fixture()
    def step(self):
        return 1.0e-3

    @pytest.fixture()
    def cell_length(self):
        return 12.0

    def compute_output_changes(self, score_network, radial_cutoff, cell_length, step):
        outputs = []
        with torch.no_grad():
            for distance in torch.arange(radial_cutoff - 0.2, radial_cutoff + 0.2, step):
                cartesian_positions = torch.tensor(
                    [[4.0, 4.0, 4.0], [4.0 + distance, 4.0, 4.0], [4.0, 5.5, 4.0], [5.5 + distance, 4.0, 4.0]]
                )
                relative_coordinates = (cartesian_positions / cell_length).unsqueeze(0)
                scores = score_network(make_batch(relative_coordinates, torch.tensor([[cell_length] * 3])))
                outputs.append(torch.cat([scores.X.flatten(), scores.A[torch.isfinite(scores.A)]]))
        outputs = torch.stack(outputs)
        return (outputs[1:] - outputs[:-1]).abs().max(dim=1).values

    def test_smooth_cutoff_is_continuous(self, radial_cutoff, cell_length, step):
        score_network = make_score_network(radial_cutoff=radial_cutoff, smooth_cutoff=True)
        output_changes = self.compute_output_changes(score_network, radial_cutoff, cell_length, step)
        assert output_changes.max() < 5.0 * step

    def test_hard_cutoff_is_discontinuous(self, radial_cutoff, cell_length, step):
        score_network = make_score_network(radial_cutoff=radial_cutoff, smooth_cutoff=False)
        output_changes = self.compute_output_changes(score_network, radial_cutoff, cell_length, step)
        assert output_changes.max() > 1.0e-3

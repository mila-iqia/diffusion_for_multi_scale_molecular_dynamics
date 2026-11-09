import itertools

import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn_utils import (
    build_graph_periodic_edges_batch, build_periodic_shift_grid,
    get_edges_with_radial_cutoff, unsorted_segment_mean, unsorted_segment_sum)


@pytest.fixture()
def num_messages():
    return 15


@pytest.fixture()
def num_ids():
    return 3


@pytest.fixture()
def message_ids(num_messages, num_ids):
    return torch.randint(low=0, high=num_ids, size=(num_messages,))


@pytest.fixture()
def num_message_features():
    return 2


@pytest.fixture()
def messages(num_messages, num_message_features):
    return torch.randn(num_messages, num_message_features)


SI_BOND_LENGTH_ANG = 2.36


@pytest.mark.parametrize("box_size", [10.0, 20.0])
def test_get_edges_with_radial_cutoff_distances_are_cell_size_independent(box_size):
    """Radial-cutoff edge distances should be the same regardless of box size."""
    radial_cutoff = 2.5
    reduced_coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                                         [SI_BOND_LENGTH_ANG / box_size, 0.0, 0.0]]])
    unit_cell = torch.diag(torch.tensor([box_size, box_size, box_size])).unsqueeze(0)

    edges = get_edges_with_radial_cutoff(reduced_coordinates, unit_cell,
                                         radial_cutoff=radial_cutoff, spatial_dimension=3)

    distances = edges[:, 2]
    torch.testing.assert_close(distances, torch.full_like(distances, SI_BOND_LENGTH_ANG))


def test_get_edges_with_radial_cutoff_padded_atoms_match_unpadded():
    """Edges from a padded batch should match those from the equivalent unpadded batch.

    Padded atom slots have NaN reduced coordinates. This test verifies that:
      - the output contains no NaN distances,
      - no edge index points to a padded slot,
      - the edges and distances are identical to the unpadded reference.
    """
    radial_cutoff = 2.5
    box_size = 10.0
    num_real_atoms = 2
    num_padded_atoms = 3

    real_reduced = torch.tensor([[0.0, 0.0, 0.0],
                                 [SI_BOND_LENGTH_ANG / box_size, 0.0, 0.0]])
    unit_cell = torch.diag(torch.tensor([box_size, box_size, box_size])).unsqueeze(0)

    # Reference: unpadded batch with only the two real atoms.
    reference_edges = get_edges_with_radial_cutoff(
        real_reduced.unsqueeze(0), unit_cell, radial_cutoff=radial_cutoff, spatial_dimension=3
    )

    # Padded batch: real atoms followed by NaN-position padding slots.
    padding = torch.full((num_padded_atoms, 3), float('nan'))
    padded_reduced = torch.cat([real_reduced, padding], dim=0).unsqueeze(0)
    natoms = torch.tensor([num_real_atoms])

    padded_edges = get_edges_with_radial_cutoff(
        padded_reduced, unit_cell, radial_cutoff=radial_cutoff, spatial_dimension=3, natoms=natoms
    )

    assert not padded_edges[:, 2].isnan().any(), "Output distances must not contain NaN"
    assert (padded_edges[:, :2] < num_real_atoms).all(), "No edge should point to a padded atom slot"
    torch.testing.assert_close(padded_edges, reference_edges)


def test_unsorted_segment_sum(
    num_messages, num_ids, message_ids, num_message_features, messages
):
    expected_message_sums = torch.zeros(num_ids, num_message_features)
    for i in range(num_messages):
        m_id = message_ids[i]
        message = messages[i]
        expected_message_sums[m_id] += message

    message_summed = unsorted_segment_sum(messages, message_ids, num_ids)
    assert message_summed.size() == torch.Size((num_ids, num_message_features))
    assert torch.allclose(message_summed, expected_message_sums)


def test_unsorted_segment_mean(
    num_messages, num_ids, message_ids, num_message_features, messages
):
    expected_message_sums = torch.zeros(num_ids, num_message_features)
    expected_counts = torch.zeros(num_ids, 1)
    for i in range(num_messages):
        m_id = message_ids[i]
        message = messages[i]
        expected_message_sums[m_id] += message
        expected_counts[m_id] += 1
    expected_message_average = expected_message_sums / torch.maximum(
        expected_counts, torch.ones_like(expected_counts)
    )

    message_averaged = unsorted_segment_mean(messages, message_ids, num_ids)
    assert message_averaged.size() == torch.Size((num_ids, num_message_features))
    assert torch.allclose(message_averaged, expected_message_average)


@pytest.mark.parametrize("cell_length, radial_cutoff, expected_n_per_axis", [
    (4.0, 2.0, 3),   # radial_cutoff == L/2: single-image regime, 3 candidates per axis (-1, 0, 1)
    (2.0, 3.0, 5),   # radial_cutoff == 1.5*L: 5 candidates per axis (-2, ..., 2)
])
def test_build_periodic_shift_grid_size(cell_length, radial_cutoff, expected_n_per_axis):
    cell_lengths = torch.tensor([cell_length, cell_length, cell_length])
    shift_grid = build_periodic_shift_grid(cell_lengths, radial_cutoff)

    assert shift_grid.shape == (expected_n_per_axis ** 3, 3)
    assert (shift_grid == 0).all(dim=-1).any(), "the zero shift must always be included"

    max_shift = (expected_n_per_axis - 1) // 2 * cell_length
    assert shift_grid.min().item() == pytest.approx(-max_shift)
    assert shift_grid.max().item() == pytest.approx(max_shift)


def test_build_graph_periodic_edges_single_image_matches_minimum_image():
    """At radial_cutoff < L/2, only the minimum image should ever qualify -- exactly like today's
    single-image edges."""
    cell_lengths = torch.tensor([10.0, 10.0, 10.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [SI_BOND_LENGTH_ANG, 0.0, 0.0]])
    radial_cutoff = 4.0

    edges = build_graph_periodic_edges_batch(positions.unsqueeze(0), cell_lengths.unsqueeze(0), radial_cutoff)

    assert edges.shape[0] == 2  # (0 -> 1) and (1 -> 0), one edge each, no other images
    torch.testing.assert_close(edges[:, 2:], torch.zeros(2, 3))  # far from any cell boundary
    distances = (positions[edges[:, 0].long()] - positions[edges[:, 1].long()] - edges[:, 2:]).norm(dim=-1)
    torch.testing.assert_close(distances, torch.full_like(distances, SI_BOND_LENGTH_ANG))


def test_build_graph_periodic_edges_diagonal_shift():
    """The nearest image can require a nonzero shift in more than one axis at once."""
    cell_lengths = torch.tensor([4.0, 4.0, 4.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [3.0, 3.0, 0.0]])
    radial_cutoff = 1.8  # comfortably above the diagonal image's sqrt(2) Ang, well below any other image

    edges = build_graph_periodic_edges_batch(positions.unsqueeze(0), cell_lengths.unsqueeze(0), radial_cutoff)

    forward_edge = edges[(edges[:, 0] == 0) & (edges[:, 1] == 1)]
    assert forward_edge.shape[0] == 1
    shift = forward_edge[0, 2:]
    torch.testing.assert_close(shift, torch.tensor([-4.0, -4.0, 0.0]))
    assert (shift != 0).sum() == 2, "shift should be nonzero along two axes at once"

    displacement = positions[0] - positions[1] - shift
    torch.testing.assert_close(displacement.norm(), torch.tensor(2.0).sqrt())


def test_build_graph_periodic_edges_multiple_images_for_one_pair():
    """A single (i, j) pair can have more than one periodic image within radial_cutoff."""
    cell_lengths = torch.tensor([3.0, 3.0, 3.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    radial_cutoff = 1.7  # includes both the 1.4 Ang and 1.6 Ang images of atom 1, excludes farther ones

    edges = build_graph_periodic_edges_batch(positions.unsqueeze(0), cell_lengths.unsqueeze(0), radial_cutoff)

    forward_edges = edges[(edges[:, 0] == 0) & (edges[:, 1] == 1)]
    assert forward_edges.shape[0] == 2
    distances = (positions[0] - positions[1] - forward_edges[:, 2:]).norm(dim=-1).sort().values
    torch.testing.assert_close(distances, torch.tensor([1.4, 1.6]))


def test_build_graph_periodic_edges_self_image_included_when_within_cutoff():
    """An atom's own periodic image is a valid edge when the cell is small relative to radial_cutoff."""
    cell_lengths = torch.tensor([5.0, 5.0, 5.0])
    positions = torch.tensor([[0.0, 0.0, 0.0]])
    radial_cutoff = 5.2  # just above the cell length: the atom's own image, one cell away, is included

    edges = build_graph_periodic_edges_batch(positions.unsqueeze(0), cell_lengths.unsqueeze(0), radial_cutoff)

    assert edges.shape[0] == 6  # one image in each of +-x, +-y, +-z
    distances = edges[:, 2:].norm(dim=-1)
    torch.testing.assert_close(distances, torch.full_like(distances, 5.0))


def test_build_graph_periodic_edges_excludes_trivial_self_edge():
    """The zero-distance (i == i, shift == 0) self-edge must never be returned."""
    cell_lengths = torch.tensor([5.0, 5.0, 5.0])
    positions = torch.tensor([[0.0, 0.0, 0.0]])
    radial_cutoff = 1.0  # well below the cell length: no periodic image is close enough either

    edges = build_graph_periodic_edges_batch(positions.unsqueeze(0), cell_lengths.unsqueeze(0), radial_cutoff)
    assert edges.shape[0] == 0


def test_build_graph_periodic_edges_non_cubic_cell_axes_independent():
    """Each axis' periodicity is handled independently for a non-cubic orthogonal cell."""
    cell_lengths = torch.tensor([3.0, 10.0, 10.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    radial_cutoff = 1.7  # only x is small enough relative to radial_cutoff for a second image to matter

    edges = build_graph_periodic_edges_batch(positions.unsqueeze(0), cell_lengths.unsqueeze(0), radial_cutoff)

    forward_edges = edges[(edges[:, 0] == 0) & (edges[:, 1] == 1)]
    assert forward_edges.shape[0] == 2
    assert (forward_edges[:, 3] == 0).all() and (forward_edges[:, 4] == 0).all(), "y/z shift must stay zero"


def _brute_force_periodic_edges(cartesian_positions, cell_lengths, radial_cutoff, shift_search_range=4):
    """Independent, unvectorized reference used to cross-check build_graph_periodic_edges_batch."""
    n_atoms, spatial_dimension = cartesian_positions.shape
    shift_values = range(-shift_search_range, shift_search_range + 1)
    edges = set()
    for i in range(n_atoms):
        for j in range(n_atoms):
            for shift in itertools.product(shift_values, repeat=spatial_dimension):
                if i == j and shift == (0,) * spatial_dimension:
                    continue
                shift_cartesian = torch.tensor(shift, dtype=torch.float) * cell_lengths
                displacement = cartesian_positions[i] - cartesian_positions[j] - shift_cartesian
                distance = displacement.norm().item()
                if distance <= radial_cutoff:
                    edges.add((i, j, tuple(round(s, 4) for s in shift_cartesian.tolist()), round(distance, 4)))
    return edges


def _edges_tensor_to_set(edges, cartesian_positions):
    result = set()
    for entry in edges.tolist():
        i, j = int(entry[0]), int(entry[1])
        shift = torch.tensor(entry[2:])
        distance = (cartesian_positions[i] - cartesian_positions[j] - shift).norm().item()
        result.add((i, j, tuple(round(s, 4) for s in shift.tolist()), round(distance, 4)))
    return result


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_build_graph_periodic_edges_matches_brute_force_reference(seed):
    """Cross-check the vectorized implementation against an independent, unvectorized reference."""
    generator = torch.Generator().manual_seed(seed)
    n_atoms = 5
    cell_lengths = torch.tensor([2.5, 3.0, 4.0])
    positions = torch.rand(n_atoms, 3, generator=generator) * cell_lengths
    radial_cutoff = 3.0  # comparable to the smallest cell length: forces multi-image edges

    edges = build_graph_periodic_edges_batch(positions.unsqueeze(0), cell_lengths.unsqueeze(0), radial_cutoff)
    actual = _edges_tensor_to_set(edges, positions)
    assert len(actual) == edges.shape[0], "no two edges should collapse to the same (i, j, shift, distance)"

    expected = _brute_force_periodic_edges(positions, cell_lengths, radial_cutoff)
    assert actual == expected


@pytest.mark.parametrize("bad_cell_length", [0.0, -0.3])
def test_build_graph_periodic_edges_rejects_non_positive_cell_lengths(bad_cell_length):
    cell_lengths = torch.tensor([[10.0, 10.0, 10.0], [10.0, bad_cell_length, 10.0]])
    positions = torch.rand(2, 4, 3) * 5.0
    with pytest.raises(RuntimeError, match="strictly positive"):
        build_graph_periodic_edges_batch(positions, cell_lengths, radial_cutoff=3.0)

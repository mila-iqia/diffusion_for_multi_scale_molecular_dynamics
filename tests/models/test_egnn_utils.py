import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.egnn_utils import (
    get_edges_batch, get_edges_with_radial_cutoff, unsorted_segment_mean,
    unsorted_segment_sum)


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
def test_get_edges_batch_distances_are_cell_size_independent(box_size):
    """Fully-connected edge distances should be the same regardless of box size."""
    reduced_coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                                         [SI_BOND_LENGTH_ANG / box_size, 0.0, 0.0]]])
    unit_cell = torch.diag(torch.tensor([box_size, box_size, box_size])).unsqueeze(0)

    edges = get_edges_batch(n_nodes=2, batch_size=1,
                            reduced_coordinates=reduced_coordinates,
                            unit_cell=unit_cell)

    distances = edges[:, 2]
    torch.testing.assert_close(distances, torch.full_like(distances, SI_BOND_LENGTH_ANG))


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

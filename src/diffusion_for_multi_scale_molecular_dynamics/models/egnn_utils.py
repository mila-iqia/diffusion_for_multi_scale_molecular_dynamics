from typing import Optional

import torch

from diffusion_for_multi_scale_molecular_dynamics.models.mace_utils import \
    get_adj_matrix
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_positions_from_coordinates


def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Sum all the elements in data by their ids.

    For example, data could be messages from atoms j to i. We want to sum all messages going to i, i.e. sum all elements
    in the message tensor that are going to i. This is indicated by the segment_ids input.

    Args:
        data: tensor to aggregate. Size is
            (number of elements to aggregate (e.g. number of edges in the message example), number of features)
        segment_ids: ids of each element in data (e.g. messages going to node i in the message example)
        num_segments: number of distinct elements in the data tensor

    Returns:
        tensor with the sum of data elements over ids. size: (num_segments, number of features)
    """
    result_shape = (num_segments, data.size(1))  # output size
    result = torch.zeros(result_shape).to(
        data
    )  # results starting as zeros - same dtype and device as data
    # tensor size manipulation to use a scatter_add operation
    # from (number of elements) to (number of elements, number of features) i.e. same size as data
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    # segment_ids needs to have the same size as data for the backward pass to go through
    # see https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_add_.html#torch.Tensor.scatter_add_
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Average all the elements in data by their ids.

    For example, data could be messages from atoms j to i. We want to average all messages going to i
    i.e. average all elements in the message tensor that are going to i. This is indicated by the segment_ids input.

    Args:
        data: tensor to aggregate. Size is
            (number of elements to aggregate (e.g. number of edges in the message example), number of features)
        segment_ids: ids of each element in data (e.g. messages going to node i in the message example)
        num_segments: number of distinct elements in the data tensor

    Returns:
        tensor with the average of data elements over ids. size: (num_segments, number of features)
    """
    result_shape = (num_segments, data.size(1))  # output size
    segment_ids = segment_ids.unsqueeze(-1).expand(
        -1, data.size(1)
    )  # tensor size manipulation for the backward pass
    result = torch.zeros(result_shape).to(data)  # sum the component
    count = torch.zeros(result_shape).to(
        data
    )  # count the number of data elements for each id to take the average
    result.scatter_add_(0, segment_ids, data)  # sum the data elements
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(
        min=1
    )  # avoid dividing by zeros by clamping the counts to be at least 1


def get_edges(n_nodes: int) -> torch.Tensor:
    """Get a tensor of edges for a fully connected graph.

    Args:
        n_nodes: number of nodes

    Returns:
        LongTensor of shape [n_nodes * (n_nodes - 1), 2] with (src, dst) pairs
    """
    indices = torch.arange(n_nodes)
    src = indices.repeat_interleave(n_nodes)
    dst = indices.repeat(n_nodes)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=1)


def get_edges_batch(
    n_nodes: int,
    batch_size: int,
    reduced_coordinates: torch.Tensor,
    unit_cell: torch.Tensor,
    natoms: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Get edges batch with cartesian distances.

    Create a tensor for all edges in a fully connected graph repeated batch_size times, with the
    minimum-image cartesian distance (Angstrom) as a third column.

    Args:
        n_nodes: number of nodes per graph (including any padding)
        batch_size: number of graphs
        reduced_coordinates: atomic positions in fractional coordinates [batch_size, n_nodes, spatial_dimension]
        unit_cell: unit cell vectors [batch_size, spatial_dimension, spatial_dimension]
        natoms: optional tensor of shape [batch_size] with the real (non-padded) atom count per sample.
            If provided, edges involving padded atom slots are excluded.

    Returns:
        float tensor of size [number of edges, 3], where the first two columns are edge indices
        (src, dst) and the third is the cartesian distance in Angstrom.
    """
    edges = get_edges(n_nodes)
    if batch_size > 1:
        num_edges = edges.shape[0]
        offsets = torch.arange(batch_size, dtype=torch.long).repeat_interleave(num_edges) * n_nodes
        edges = edges.repeat(batch_size, 1) + offsets.unsqueeze(1)
    edges = edges.to(reduced_coordinates.device)

    flat_reduced = reduced_coordinates.view(-1, reduced_coordinates.shape[-1])
    src, dst = edges[:, 0], edges[:, 1]
    batch_idx = src // n_nodes

    if natoms is not None:
        real_edge_mask = (
            (src % n_nodes < natoms[batch_idx]) & (dst % n_nodes < natoms[batch_idx])
        )
        edges = edges[real_edge_mask]
        src, dst = edges[:, 0], edges[:, 1]
        batch_idx = src // n_nodes

    delta_x = flat_reduced[dst] - flat_reduced[src]
    delta_x = delta_x - torch.round(delta_x)  # minimum image convention in reduced space
    delta_r = torch.bmm(delta_x.unsqueeze(1), unit_cell[batch_idx]).squeeze(1)
    edge_distances = delta_r.norm(dim=-1)

    return torch.cat([edges.float(), edge_distances.unsqueeze(1)], dim=1)


def get_edges_with_radial_cutoff(
    reduced_coordinates: torch.Tensor,
    unit_cell: torch.Tensor,
    radial_cutoff: float = 4.0,
    spatial_dimension: int = 3,
    natoms: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Get edges for a batch with a cutoff based on distance, including cartesian distances.

    Each (src, dst) pair with a distinct PBC shift appears as a separate edge, since each represents a
    physically distinct interaction with its own cartesian distance.

    Args:
        reduced_coordinates: batch x n_atom x spatial dimension tensor with reduced coordinates
        unit_cell: batch x spatial dimension x spatial dimension tensor with the unit cell vectors
        radial_cutoff (optional): cutoff distance in Angstrom. Defaults to 4.0
        spatial_dimension (optional): spatial dimension. Defaults to 3.
        natoms (optional): tensor of shape [batch_size] with the real (non-padded) atom count per sample.
            If provided, edges involving padded atom slots are excluded.

    Returns:
        float tensor of size [number of edges, 3], where the first two columns are edge indices (src, dst)
        and the third is the cartesian distance in Angstrom.
    """
    if natoms is not None:
        # Padded atoms have finite (non-NaN) positions that would be counted as neighbors by KeOps,
        # inflating max_number_of_neighbors and K by orders of magnitude. Set them to NaN so KeOps
        # correctly ignores them (NaN <= r² evaluates to False in IEEE 754).
        n_nodes = reduced_coordinates.shape[1]
        padded_mask = torch.arange(n_nodes, device=reduced_coordinates.device).unsqueeze(0) >= natoms.unsqueeze(1)
        reduced_coordinates = reduced_coordinates.clone()
        reduced_coordinates[padded_mask] = float('nan')

    cartesian_coordinates = get_positions_from_coordinates(reduced_coordinates, unit_cell)
    adj_matrix, _, _, _, squared_distances = get_adj_matrix(
        cartesian_coordinates, unit_cell, radial_cutoff, spatial_dimension
    )

    if natoms is not None:
        src, dst = adj_matrix[0], adj_matrix[1]
        batch_idx = src // n_nodes
        real_edge_mask = (
            (src % n_nodes < natoms[batch_idx]) & (dst % n_nodes < natoms[batch_idx])
        )
        adj_matrix = adj_matrix[:, real_edge_mask]
        squared_distances = squared_distances[real_edge_mask]

    edge_distances = squared_distances.sqrt()

    # MACE adj calculations returns a (2, n_edges) tensor and EGNN expects a (n_edges, 2) tensor
    adj_matrix = adj_matrix.transpose(0, 1)

    return torch.cat([adj_matrix.float(), edge_distances.unsqueeze(1)], dim=1)

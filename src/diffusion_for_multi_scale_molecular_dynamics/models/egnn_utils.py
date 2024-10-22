from typing import List

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


def get_edges(n_nodes: int) -> List[List[int]]:
    """Get a list of nodes for a fully connected graph.

    Args:
        n_nodes: number of nodes

    Returns:
        list of n_nodes * (n_nodes - 1) connections (list of 2 integers)
    """
    return [[x, y] for x in range(n_nodes) for y in range(n_nodes) if x != y]


def get_edges_batch(n_nodes: int, batch_size: int) -> torch.Tensor:
    """Get edges batch.

    Create a tensor indicating all the source/destination nodes in a fully connected graph repeated batch_size times.

    Args:
        n_nodes: number of nodes in a graph
        batch_size: number of graphs

    Returns:
        long tensor of size [number of edges = batch_size * n_nodes * (n_nodes - 1), 2]
    """
    edges = get_edges(n_nodes)
    edges = torch.LongTensor(edges)
    if batch_size > 1:
        all_edges = []
        for i in range(batch_size):
            all_edges.append(edges + n_nodes * i)
        edges = torch.cat(all_edges)
    return edges


def get_edges_with_radial_cutoff(
    relative_coordinates: torch.Tensor,
    unit_cell: torch.Tensor,
    radial_cutoff: float = 4.0,
    drop_duplicate_edges: bool = True,
) -> torch.Tensor:
    """Get edges for a batch with a cutoff based on distance.

    Args:
        relative_coordinates: batch x n_atom x spatial dimension tensor with relative coordinates
        unit_cell: batch x spatial dimension x spatial dimension tensor with the unit cell vectors
        radial_cutoff (optional): cutoff distance in Angstrom. Defaults to 4.0
        drop_duplicate_edges (optional): if True, return only 1 instance of each edge. If False, return each edge
            multiple times, depending on the unit cell shift multiplicities. Defaults to True.

    Returns:
        long tensor of size [number of edges, 2] with edge indices
    """
    # get cartesian coordinates from relative coordinates
    cartesian_coordinates = get_positions_from_coordinates(
        relative_coordinates, unit_cell
    )
    adj_matrix, _, _, _ = get_adj_matrix(
        cartesian_coordinates, unit_cell, radial_cutoff
    )
    # adj_matrix is a n_edges x 2 tensor with duplicates with different shifts.
    # the uplifting in 2 x spatial_dimension manages the shifts in a natural way. This means we can ignore the shifts
    # and possibly ignore the multiplicities i.e. no need to sum twice the contribution of a neighbor that we see
    # in the unit cell and in a shifted unit cell.
    # TODO check this statement - test with and without multiplicities - just remove the duplicate drop that follows to
    # test the w/o multiplicities case
    if drop_duplicate_edges:
        adj_matrix = torch.unique(adj_matrix, dim=1)
    # MACE adj calculations returns a (2, n_edges) tensor and EGNN expects a (n_edges, 2) tensor
    adj_matrix = adj_matrix.transpose(0, 1)

    return adj_matrix

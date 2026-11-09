from typing import Optional

import torch

from diffusion_for_multi_scale_molecular_dynamics.score_network.mace_utils import \
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


def minimum_image_cartesian(raw_coord_diff: torch.Tensor, cell_lengths: torch.Tensor) -> torch.Tensor:
    """Wrap a Cartesian displacement to its minimum image, for an orthogonal cell.

    Args:
        raw_coord_diff: raw Cartesian displacement. size: number of edges, spatial_dimension
        cell_lengths: per-edge cell lengths. size: number of edges, spatial_dimension

    Returns:
        wrapped displacement, each component in [-cell_lengths/2, cell_lengths/2]. Same size as coord_diff.
    """
    number_of_cell_lengths_away = torch.round(raw_coord_diff / cell_lengths)
    wrap_correction = cell_lengths * number_of_cell_lengths_away
    return raw_coord_diff - wrap_correction


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


def _build_periodic_shift_grid_integer(cell_lengths: torch.Tensor, radial_cutoff: float) -> torch.Tensor:
    """Build every candidate periodic shift, in units of whole cells, to search around a minimum image.

    Args:
        cell_lengths: orthogonal cell side lengths. size: (spatial_dimension,)
        radial_cutoff: cutoff distance, in Angstrom.

    Returns:
        float tensor of size (n_shifts, spatial_dimension), each row a whole number of cells per axis.
    """
    device = cell_lengths.device
    spatial_dimension = cell_lengths.shape[0]
    max_shift_per_axis = torch.ceil(radial_cutoff / cell_lengths + 0.5).long()
    shift_ranges = [torch.arange(-n.item(), n.item() + 1, device=device) for n in max_shift_per_axis]
    return torch.cartesian_prod(*shift_ranges).float().view(-1, spatial_dimension)


def build_periodic_shift_grid(cell_lengths: torch.Tensor, radial_cutoff: float) -> torch.Tensor:
    """Build every candidate periodic shift, in Angstrom, to search around a minimum image.

    Args:
        cell_lengths: orthogonal cell side lengths. size: (spatial_dimension,)
        radial_cutoff: cutoff distance, in Angstrom.

    Returns:
        float tensor of size (n_shifts, spatial_dimension), the cartesian shift of each candidate.
    """
    return _build_periodic_shift_grid_integer(cell_lengths, radial_cutoff) * cell_lengths


def build_graph_periodic_edges_batch(
    cartesian_positions: torch.Tensor,
    cell_lengths: torch.Tensor,
    radial_cutoff: float,
    natoms: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build every edge within radial_cutoff, including periodic images, for a batch of orthogonal cells.

    Fully vectorized over the batch: the integer shift search range is shared across the batch, sized
    for the smallest cell length present (safe for every sample -- a larger cell in the same batch just
    has a few extra candidate shifts filtered out). Each sample's shifts are then scaled by its own
    cell_lengths, so the returned cartesian shifts are always exact for that sample.

    Args:
        cartesian_positions: atomic positions. size: (batch_size, max_atoms, spatial_dimension)
        cell_lengths: per-sample orthogonal cell side lengths. size: (batch_size, spatial_dimension)
        radial_cutoff: cutoff distance, in Angstrom.
        natoms: optional real (non-padded) atom count per sample. size: (batch_size,).
            If provided, edges involving padded atom slots are excluded.

    Returns:
        float tensor of size (n_edges, 2 + spatial_dimension). The first two are node index i and j,
        into the flattened (batch_size * max_atoms) node numbering. The rest is the cartesian shift
        (Angstrom), in all 3 directions, used to construct the edge.

    Raises:
        RuntimeError: if a cell length is not strictly positive.
    """
    if not (cell_lengths > 0).all():
        raise RuntimeError(
            f"All cell lengths must be strictly positive to build periodic edges. "
            f"Got a minimum cell length of {cell_lengths.min().item():.4f} Angstrom."
        )

    batch_size, max_atoms, spatial_dimension = cartesian_positions.shape
    device = cartesian_positions.device

    if natoms is not None:
        padded_mask = torch.arange(max_atoms, device=device).unsqueeze(0) >= natoms.unsqueeze(1)
        cartesian_positions = cartesian_positions.clone()
        cartesian_positions[padded_mask] = float('nan')

    raw_diff = cartesian_positions.unsqueeze(2) - cartesian_positions.unsqueeze(1)  # (batch, max_atoms, max_atoms, d)

    # position_displacement puts raw_diff within [-L/2,L/2], per sample (each sample has its own cell_lengths).
    cell_lengths_per_pair = cell_lengths.view(batch_size, 1, 1, spatial_dimension)
    position_displacement = cell_lengths_per_pair * torch.round(raw_diff / cell_lengths_per_pair)

    # periodic image search range sized for the smallest cell in the batch.
    smallest_cell_lengths = cell_lengths.min(dim=0).values
    shift_grid_integer = _build_periodic_shift_grid_integer(smallest_cell_lengths, radial_cutoff)  # (n_shifts, d)
    n_shifts = shift_grid_integer.shape[0]
    grid_shift_cartesian = shift_grid_integer.view(1, n_shifts, spatial_dimension) * cell_lengths.view(
        batch_size, 1, spatial_dimension
    )  # (batch_size, n_shifts, d)

    # total_shift considers every possible image cell that has volume within rcut
    total_shift = position_displacement.unsqueeze(3) + grid_shift_cartesian.view(
        batch_size, 1, 1, n_shifts, spatial_dimension
    )

    # all considered position_diff including images
    candidate_diff = raw_diff.unsqueeze(3) - total_shift  # (batch, max_atoms, max_atoms, n_shifts, d)
    squared_distances = (candidate_diff ** 2).sum(dim=-1)

    # masking to only keep valid edges
    within_cutoff = squared_distances <= radial_cutoff ** 2  # NaN <= radial_cutoff**2 is False: excludes padding
    zero_shift_index = (shift_grid_integer == 0).all(dim=-1).nonzero(as_tuple=True)[0].item()
    self_edge_mask = torch.eye(max_atoms, dtype=torch.bool, device=device).view(1, max_atoms, max_atoms, 1)
    self_edge_mask = self_edge_mask & (torch.arange(n_shifts, device=device) == zero_shift_index)
    within_cutoff = within_cutoff & ~self_edge_mask

    batch_idx, row, col, shift_index = within_cutoff.nonzero(as_tuple=True)
    edges = torch.stack([row + batch_idx * max_atoms, col + batch_idx * max_atoms], dim=1).float()
    shifts = total_shift[batch_idx, row, col, shift_index]
    return torch.cat([edges, shifts], dim=1)

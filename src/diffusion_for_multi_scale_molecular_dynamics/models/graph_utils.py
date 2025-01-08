from typing import Tuple

import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import (
    get_periodic_adjacency_information,
    shift_adjacency_matrix_indices_for_graph_batching)


def get_adj_matrix(
    positions: torch.Tensor, basis_vectors: torch.Tensor, radial_cutoff: float = 4.0, spatial_dimension: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the adjacency and shift matrices.

    Args:
         positions : atomic positions, assumed to be within the unit cell, in Euclidean coordinates.
                               Dimension [batch_size, max_number_of_atoms, 3]
        basis_vectors : vectors that define the unit cell, (a1, a2, a3). The basis vectors are assumed
                        to be vertically stacked, namely
                                            [-- a1 --]
                                            [-- a2 --]
                                            [-- a3 --]
                        Dimension [batch_size, 3, 3].
        radial_cutoff : largest distance between neighbors.

    Returns:
        adjacency matrix: The (src, dst) node indices, as a [2, num_edge] tensor,
        shift matrix: The lattice vector shifts between source and destination, as a [num_edge, 3] tensor
        batch_indices: for each node, this indicates which batch item it originally belonged to.
        number_of_edges: for each element in the batch, how many edges belong to it
    """
    batch_size, number_of_atoms, spatial_dimensions = positions.shape

    adjacency_info = get_periodic_adjacency_information(
        positions, basis_vectors, radial_cutoff, spatial_dimension
    )

    # The indices in the adjacency matrix must be shifted to account for the batching
    # of multiple distinct structures into a single disconnected graph.
    adjacency_matrix = adjacency_info.adjacency_matrix
    number_of_edges = adjacency_info.number_of_edges
    shifted_adjacency_matrix = shift_adjacency_matrix_indices_for_graph_batching(
        adjacency_matrix, number_of_edges, number_of_atoms
    )
    shifts = adjacency_info.shifts
    batch_indices = adjacency_info.node_batch_indices

    number_of_edges = adjacency_info.number_of_edges

    return shifted_adjacency_matrix, shifts, batch_indices, number_of_edges

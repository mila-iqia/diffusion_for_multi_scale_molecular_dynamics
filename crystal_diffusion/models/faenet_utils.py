from typing import AnyStr, Dict, Tuple

import torch
from torch_geometric.data import Data

from crystal_diffusion.namespace import (NOISE, NOISY_CARTESIAN_POSITIONS,
                                         NOISY_RELATIVE_COORDINATES, UNIT_CELL)
from crystal_diffusion.utils.neighbors import (
    get_periodic_adjacency_information,
    shift_adjacency_matrix_indices_for_graph_batching)


def get_adj_matrix(positions: torch.Tensor,
                   basis_vectors: torch.Tensor,
                   radial_cutoff: float = 4.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    adjacency_info = get_periodic_adjacency_information(positions, basis_vectors, radial_cutoff)

    # The indices in the adjacency matrix must be shifted to account for the batching
    # of multiple distinct structures into a single disconnected graph.
    adjacency_matrix = adjacency_info.adjacency_matrix
    number_of_edges = adjacency_info.number_of_edges
    shifted_adjacency_matrix = shift_adjacency_matrix_indices_for_graph_batching(adjacency_matrix,
                                                                                 number_of_edges,
                                                                                 number_of_atoms)
    shifts = adjacency_info.shifts
    batch_indices = adjacency_info.node_batch_indices

    number_of_edges = adjacency_info.number_of_edges

    return shifted_adjacency_matrix, shifts, batch_indices, number_of_edges


def input_to_faenet(x: Dict[AnyStr, torch.Tensor], radial_cutoff: float) -> Data:
    """Convert score network input to MACE input.

    Args:
        x: score network input dictionary
        radial_cutoff : largest distance between neighbors.

    Returns:
        pytorch-geometric graph data compatible with MACE forward
    """
    noisy_cartesian_positions = x[NOISY_CARTESIAN_POSITIONS]
    cell = x[UNIT_CELL]  # batch, spatial_dimension, spatial_dimension

    batch_size, n_atom_per_graph, spatial_dimension = noisy_cartesian_positions.shape
    device = noisy_cartesian_positions.device
    adj_matrix, shift_matrix, batch_tensor, edge_per_batch = get_adj_matrix(positions=noisy_cartesian_positions,
                                                                            basis_vectors=cell,
                                                                            radial_cutoff=radial_cutoff)

    sigmas = x[NOISE].unsqueeze(1).repeat(1, n_atom_per_graph, 1)  # batch, n_atom, 1
    # create the pytorch-geometric graph
    # TODO do not hard-code Si
    graph_data = Data(edge_index=adj_matrix,
                      neighbors=edge_per_batch,  # number of edges for each batch
                      pos=x[NOISY_RELATIVE_COORDINATES].view(-1, spatial_dimension),  # reduced positions N, 3
                      atomic_numbers=(torch.ones(batch_size * n_atom_per_graph) * 14).long().to(device),
                      batch=batch_tensor.to(device),
                      cell_offsets=shift_matrix,
                      cell=cell,   # batch, spatial dimension, spatial dimension,
                      sigma=sigmas.view(-1, 1).to(device)  # batch_size * n_atom_per_graph
                      )

    return graph_data

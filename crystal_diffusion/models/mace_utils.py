from typing import AnyStr, Dict, Tuple

import torch
from torch_geometric.data import Data

from crystal_diffusion.utils.neighbors import (
    get_periodic_adjacency_information,
    shift_adjacency_matrix_indices_for_graph_batching)


def get_adj_matrix(positions: torch.Tensor,
                   basis_vectors: torch.Tensor,
                   radial_cutoff: float = 4.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    return shifted_adjacency_matrix, shifts, batch_indices


def input_to_mace(x: Dict[AnyStr, torch.Tensor], unit_cell_key: str, radial_cutoff: float) -> Data:
    """Convert score network input to MACE input.

    Args:
        x: score network input dictionary
        unit_cell_key: keyword argument to find the cell definition
        radial_cutoff : largest distance between neighbors.

    Returns:
        pytorch-geometric graph data compatible with MACE forward
    """
    batch_size = x['abs_positions'].size(0)
    cell = x[unit_cell_key]  # batch, spatial_dimension, spatial_dimension
    n_atom_per_graph = x['abs_positions'].size(1)
    device = x['abs_positions'].device
    # TODO : make the radial cut-off available here
    adj_matrix, shift_matrix, batch_tensor = get_adj_matrix(positions=x['abs_positions'],
                                                            basis_vectors=cell,
                                                            radial_cutoff=radial_cutoff)
    # node features are int corresponding to atom type
    node_attrs = torch.ones(batch_size * n_atom_per_graph, 1)  # TODO handle different type of atoms
    positions = x['abs_positions'].view(-1, x['abs_positions'].size(-1))  # [batchsize * natoms, spatial dimension]
    # pointer tensor that yields the first node index for each batch - this is a fixed tensor in our case
    ptr = torch.arange(0, n_atom_per_graph * batch_size + 1, step=n_atom_per_graph)  # 0, natoms, 2 * natoms, ...

    cell = cell.view(-1, cell.size(-1))  # batch * spatial_dimension, spatial_dimension
    # create the pytorch-geometric graph
    graph_data = Data(edge_index=adj_matrix,
                      node_attrs=node_attrs.to(device),
                      positions=positions,
                      ptr=ptr.to(device),
                      batch=batch_tensor.to(device),
                      shifts=shift_matrix,
                      cell=cell
                      )
    return graph_data

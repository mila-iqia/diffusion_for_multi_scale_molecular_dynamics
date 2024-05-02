from typing import AnyStr, Dict, Tuple

import torch
from torch_geometric.data import Data


def get_adj_matrix(num_edge: int, cell_size: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create the adjacency and shift matrices.

    Random matrices for now. Placeholder.

    Args:
        num_edge: number of edges to place in the graph
        cell_size: size of the unit cell in Angstrom to get shift in Angstrom

    Returns:
        adjacency matrix as a [2, num_edge] tensor, and shift matrix as a [num_edge, 3] tensor
    """
    adj = torch.randint(0, 2, size=(2, num_edge))
    shift = torch.randint(-1, 2, size=(num_edge, 3)) * cell_size
    return adj, shift


def input_to_mace(x: Dict[AnyStr, torch.Tensor], unit_cell_key: str) -> Data:
    """Convert score network input to MACE input.

    Args:
        x: score network input dictionary
        unit_cell_key: keyword argument to find the cell definition

    Returns:
        pytorch-geometric graph data compatible with MACE forward
    """
    batchsize = x['abs_positions'].size(0)
    n_atom_per_graph = x['abs_positions'].size(1)
    device = x['abs_positions'].device
    adj_matrix, shift_matrix = get_adj_matrix(torch.randint(2, (batchsize * n_atom_per_graph) ** 2))  # TODO placeholder
    # node features are int corresponding to atom type
    node_attrs = torch.ones(batchsize, 1)  # TODO handle different type of atoms
    positions = x['abs_positions'].view(-1, x['abs_positions'].size(-1))  # [batchsize * natoms, spatial dimension]
    # pointer tensor that yields the first node index for each batch - this is a fixed tensor in our case
    ptr = torch.arange(0, n_atom_per_graph * batchsize + 1, step=n_atom_per_graph)  # 0, natoms, 2 * natoms, ...
    # batch tensor maps a node to a batch index - in our case, this would be 0, 0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ...
    batch_tensor = torch.repeat_interleave(torch.arange(0, batchsize), n_atom_per_graph)
    cell = x[unit_cell_key]  # batch, spatial_dimension, spatial_dimension
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

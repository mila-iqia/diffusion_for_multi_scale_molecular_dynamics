import os
from typing import AnyStr, Dict, Tuple

import torch
import urllib
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
    adj_matrix, shift_matrix, batch_tensor = get_adj_matrix(positions=x['abs_positions'],
                                                            basis_vectors=cell,
                                                            radial_cutoff=radial_cutoff)
    # node features are int corresponding to atom type
    # TODO handle different atom types
    node_attrs = torch.nn.functional.one_hot((torch.ones(batch_size * n_atom_per_graph) * 14).long(),
                                             num_classes=89).float()
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


def download_pretrained_mace(model_name: str, model_savedir_path: str) -> Tuple[torch.nn.Module, int]:
    """Download and load a pre-trained MACE network.

    Based on the mace-torch library.
    https://github.com/ACEsuit/mace/blob/main/mace/calculators/foundations_models.py

    Args:
        model_name: which model to load. Should be small, medium or large.
        model_savedir_path: path where to save the pretrained model

    Returns:
        model: the pre-trained MACE model as a torch.nn.Module
        node_feats_output_size: size of the node features embedding in the model's output
    """
    assert model_name in ["small", "medium", "large"], f"Model name should be small, medium or large. Got {model_name}"

    # from mace library code
    urls = dict(
        small=("https://tinyurl.com/46jrkm3v", 256),  # 2023-12-10-mace-128-L0_energy_epoch-249.model
        medium=("https://tinyurl.com/5yyxdm76", 640),  # 2023-12-03-mace-128-L1_epoch-199.model
        large=("https://tinyurl.com/5f5yavf3", 1280), # MACE_MPtrj_2022.9.model
    )
    checkpoint_url, node_feats_output_size = (urls.get(model_name, urls["medium"]))

    checkpoint_url_name = "".join(
        c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_"
    )
    cached_model_path = os.path.join(model_savedir_path, checkpoint_url_name)

    if not os.path.isfile(cached_model_path):
        os.makedirs(model_savedir_path, exist_ok=True)
        # download and save to disk
        _, http_msg = urllib.request.urlretrieve(checkpoint_url, cached_model_path)
        if "Content-Type: text/html" in http_msg:
            raise RuntimeError(f"Model download failed, please check the URL {checkpoint_url}")

    model = torch.load(f=cached_model_path).float()

    return model, node_feats_output_size

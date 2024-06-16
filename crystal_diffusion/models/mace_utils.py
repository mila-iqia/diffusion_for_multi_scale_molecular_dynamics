import os
import urllib
from typing import AnyStr, Dict, Tuple

import torch
from e3nn import o3
from torch_geometric.data import Data

from crystal_diffusion.namespace import NOISY_CARTESIAN_POSITIONS, UNIT_CELL
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


def input_to_mace(x: Dict[AnyStr, torch.Tensor], radial_cutoff: float) -> Data:
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
    adj_matrix, shift_matrix, batch_tensor = get_adj_matrix(positions=noisy_cartesian_positions,
                                                            basis_vectors=cell,
                                                            radial_cutoff=radial_cutoff)
    # node features are int corresponding to atom type
    # TODO handle different atom types
    node_attrs = torch.nn.functional.one_hot((torch.ones(batch_size * n_atom_per_graph) * 14).long(),
                                             num_classes=89).float()
    flat_positions = noisy_cartesian_positions.view(-1, spatial_dimension)  # [batchsize * natoms, spatial dimension]
    # pointer tensor that yields the first node index for each batch - this is a fixed tensor in our case
    ptr = torch.arange(0, n_atom_per_graph * batch_size + 1, step=n_atom_per_graph)  # 0, natoms, 2 * natoms, ...

    cell = cell.view(-1, cell.size(-1))  # batch * spatial_dimension, spatial_dimension
    # create the pytorch-geometric graph
    graph_data = Data(edge_index=adj_matrix,
                      node_attrs=node_attrs.to(device),
                      positions=flat_positions,
                      ptr=ptr.to(device),
                      batch=batch_tensor.to(device),
                      shifts=shift_matrix,
                      cell=cell
                      )
    return graph_data


def build_mace_output_nodes_irreducible_representation(hidden_irreps_string: str, num_interactions: int) -> o3.Irreps:
    """Build the mace output node irreps.

    Args:
        hidden_irreps_string : the hidden representation irreducible string.

    Returns:
        output_node_irreps: the irreducible representation of the output node features.
    """
    # By inspection of the MACE code, ie mace.modules.models.MACE, we can see that:
    #   - in the __init__ method, the irrep of the output is the 'number of interactions' times
    #     the hidden_irrep, except for the last which is just the scalar part.
    #   - there's an assumption in the MACE code that there will always be a scalar representation (0e).
    hidden_irreps = o3.Irreps(hidden_irreps_string)

    # E3NN irreps gymnastics is a bit fragile. We have to build the scalar representation explicitly
    scalar_hidden_irreps = o3.Irreps(f"{hidden_irreps[0].mul}x{hidden_irreps[0].ir}")

    total_irreps = o3.Irreps('')  # An empty irrep to start the "sum", which is really a concatenation
    for _ in range(num_interactions - 1):
        total_irreps += hidden_irreps

    total_irreps += scalar_hidden_irreps

    return total_irreps


def get_pretrained_mace_output_node_features_irreps(model_name: str) -> o3.Irreps:
    """Get pretrained MACE output node features irreps.

    Args:
        model_name : name of the pretrained model.

    Returns:
        Irreps: the irreducible representation of the concatenated output nodes.
    """
    match model_name:
        case "small":
            irreps = build_mace_output_nodes_irreducible_representation(hidden_irreps_string="128x0e",
                                                                        num_interactions=2)
        case "medium":
            irreps = build_mace_output_nodes_irreducible_representation(hidden_irreps_string="128x0e + 128x1o",
                                                                        num_interactions=2)
        case "large":
            irreps = build_mace_output_nodes_irreducible_representation(hidden_irreps_string="128x0e + 128x1o + 128x2e",
                                                                        num_interactions=2)
        case _:
            raise ValueError(f"Model name should be small, medium or large. Got {model_name}")

    return irreps


def get_pretrained_mace(model_name: str, model_savedir_path: str) -> Tuple[torch.nn.Module, int]:
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
        large=("https://tinyurl.com/5f5yavf3", 1280),  # MACE_MPtrj_2022.9.model
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


def get_normalized_irreps_permutation_indices(irreps: o3.Irreps) -> Tuple[o3.Irreps, torch.Tensor]:
    """Get normalized irreps and permutation indices.

    Args:
        irreps : Irreducible representation corresponding to the entries in data.

    Returns:
        normalized_irreps : sorted and simplified irreps.
        column_permutation_indices: indices that can rearrange the columns of a data tensor to go from irreps to
            normalized_irreps.
    """
    sorted_output = irreps.sort()
    irreps_permutation_indices = sorted_output.inv

    column_permutation_indices = []

    for idx in irreps_permutation_indices:
        slc = irreps.slices()[idx]
        irrep_indices = list(range(slc.start, slc.stop))
        column_permutation_indices.extend(irrep_indices)

    column_permutation_indices = torch.tensor(column_permutation_indices)

    sorted_irreps = sorted_output.irreps.simplify()

    return sorted_irreps, column_permutation_indices

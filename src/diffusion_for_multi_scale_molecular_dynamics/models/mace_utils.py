import os
import urllib
from typing import AnyStr, Dict, Tuple

import torch
from e3nn import o3
from torch_geometric.data import Data

from diffusion_for_multi_scale_molecular_dynamics.models.graph_utils import \
    get_adj_matrix
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    NOISY_AXL_COMPOSITION, NOISY_CARTESIAN_POSITIONS)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_noisy_axl_lattice_parameters_to_unit_cell_vectors


def input_to_mace(
    x: Dict[AnyStr, torch.Tensor], radial_cutoff: float, unit_cell_cutoff: float = 4.0
) -> Data:
    """Convert score network input to MACE input.

    Args:
        x: score network input dictionary
        radial_cutoff : largest distance between neighbors.
        unit_cell_cutoff: minimal box cutoff in Angstrom.. Useful for noisy lattice parameters. Defaults to 4.0.

    Returns:
        pytorch-geometric graph data compatible with MACE forward
    """
    noisy_cartesian_positions = x[NOISY_CARTESIAN_POSITIONS]
    batch_size, n_atom_per_graph, spatial_dimension = noisy_cartesian_positions.shape

    cell = map_noisy_axl_lattice_parameters_to_unit_cell_vectors(
        x[NOISY_AXL_COMPOSITION].L,
        min_box_size=unit_cell_cutoff,
    )
    # cell is batch, spatial_dimension, spatial_dimension

    device = noisy_cartesian_positions.device
    adj_matrix, shift_matrix, batch_tensor, _ = get_adj_matrix(
        positions=noisy_cartesian_positions,
        basis_vectors=cell,
        radial_cutoff=radial_cutoff,
    )
    # node features are int corresponding to atom type
    # TODO handle different atom types
    node_attrs = torch.nn.functional.one_hot(
        (torch.ones(batch_size * n_atom_per_graph) * 14).long(), num_classes=89
    ).to(noisy_cartesian_positions)
    flat_positions = noisy_cartesian_positions.view(
        -1, spatial_dimension
    )  # [batchsize * natoms, spatial dimension]
    # pointer tensor that yields the first node index for each batch - this is a fixed tensor in our case
    ptr = torch.arange(
        0, n_atom_per_graph * batch_size + 1, step=n_atom_per_graph
    )  # 0, natoms, 2 * natoms, ...

    cell = cell.view(-1, cell.size(-1))  # batch * spatial_dimension, spatial_dimension
    # create the pytorch-geometric graph
    graph_data = Data(
        edge_index=adj_matrix,
        node_attrs=node_attrs.to(device),
        positions=flat_positions,
        ptr=ptr.to(device),
        batch=batch_tensor.to(device),
        shifts=shift_matrix,
        cell=cell,
    )
    return graph_data


def build_mace_output_nodes_irreducible_representation(
    hidden_irreps_string: str, num_interactions: int
) -> o3.Irreps:
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

    total_irreps = o3.Irreps(
        ""
    )  # An empty irrep to start the "sum", which is really a concatenation
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
            irreps = build_mace_output_nodes_irreducible_representation(
                hidden_irreps_string="128x0e", num_interactions=2
            )
        case "medium":
            irreps = build_mace_output_nodes_irreducible_representation(
                hidden_irreps_string="128x0e + 128x1o", num_interactions=2
            )
        case "large":
            irreps = build_mace_output_nodes_irreducible_representation(
                hidden_irreps_string="128x0e + 128x1o + 128x2e", num_interactions=2
            )
        case _:
            raise ValueError(
                f"Model name should be small, medium or large. Got {model_name}"
            )

    return irreps


def get_pretrained_mace(
    model_name: str, model_savedir_path: str
) -> Tuple[torch.nn.Module, int]:
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
    assert model_name in [
        "small",
        "medium",
        "large",
    ], f"Model name should be small, medium or large. Got {model_name}"

    # from mace library code
    urls = dict(
        small=(
            "https://tinyurl.com/46jrkm3v",
            256,
        ),  # 2023-12-10-mace-128-L0_energy_epoch-249.model
        medium=(
            "https://tinyurl.com/5yyxdm76",
            640,
        ),  # 2023-12-03-mace-128-L1_epoch-199.model
        large=("https://tinyurl.com/5f5yavf3", 1280),  # MACE_MPtrj_2022.9.model
    )
    checkpoint_url, node_feats_output_size = urls.get(model_name, urls["medium"])

    checkpoint_url_name = "".join(
        c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_"
    )
    cached_model_path = os.path.join(model_savedir_path, checkpoint_url_name)

    if not os.path.isfile(cached_model_path):
        os.makedirs(model_savedir_path, exist_ok=True)
        # download and save to disk
        _, http_msg = urllib.request.urlretrieve(checkpoint_url, cached_model_path)
        if "Content-Type: text/html" in http_msg:
            raise RuntimeError(
                f"Model download failed, please check the URL {checkpoint_url}"
            )

    model = torch.load(f=cached_model_path).float()

    return model, node_feats_output_size


def get_normalized_irreps_permutation_indices(
    irreps: o3.Irreps,
) -> Tuple[o3.Irreps, torch.Tensor]:
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


def reshape_from_mace_to_e3nn(x: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """Reshape a MACE input/output tensor to a e3nn.NormActivation compatible format.

    MACE uses tensors in the 2D format (ignoring the nodes / batchsize):
        ---- l = 0 ----
        ---- l = 1 ----
        ---- l = 1 ----
        ---- l = 1 ----
        ...
    And e3nn wants a tensor in the 1D format:
        ---- l = 0 ---- ---- l= 1 ---- ---- l=2 ---- ...

    Args:
        x: tensor used by MACE. Should be of size (number of nodes, number of channels, (ell_max + 1)^2)
        irreps: o3 irreps matching the x tensor

    Returns:
        tensor of size (number of nodes, number of channels * (ell_max + 1)^2) usable by e3nn
    """
    node = x.size(0)
    x_ = []
    for ell in range(irreps.lmax + 1):
        # for example, for l=1, take indices 1, 2, 3 (in the last index) and flatten as a channel * 3 tensor
        x_l = x[:, :, (ell**2):(ell + 1) ** 2].reshape(
            node, -1
        )  # node, channel * (2l + 1)
        x_.append(x_l)
    # stack the flatten irrep tensors together
    return torch.cat(x_, dim=-1)


def reshape_from_e3nn_to_mace(x: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """Reshape a tensor in the  e3nn.NormActivation format to a MACE format.

    See reshape_from_mace_to_e3nn for an explanation of the formats

    Args:
        x: torch used by MACE. Should be of size (number of nodes, number of channels, (ell_max + 1)^2
        irreps: o3 irreps matching the x tensor

    Returns:
        tensor of size (number of nodes, number of channels * (ell_max + 1)^2) usable by e3nn
    """
    node = x.size(0)
    x_ = []
    for ell, s in enumerate(irreps.slices()):
        x_l = x[:, s].reshape(node, -1, 2 * ell + 1)
        x_.append(x_l)
    return torch.cat(x_, dim=-1)

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import \
    AXL_COMPOSITION
from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import \
    get_periodic_adjacency_information


def get_number_of_samples_with_overlaps(samples_pickle, radial_cutoff=1.0):
    """Get number of samples with overlaps."""
    data = torch.load(samples_pickle, map_location="cpu")
    compositions = data[AXL_COMPOSITION]
    relative_coordinates = compositions.X
    basis_vectors = compositions.L
    cartesian_positions = einops.einsum(
        relative_coordinates,
        basis_vectors,
        "batch natoms space1, batch space1 space -> batch natoms space",
    )
    adjacency_info = get_periodic_adjacency_information(
        cartesian_positions, basis_vectors, radial_cutoff=radial_cutoff
    )

    edge_batch_indices = adjacency_info.edge_batch_indices

    number_of_short_edges = len(edge_batch_indices) // 2
    number_of_samples_with_overlaps = len(edge_batch_indices.unique())
    return number_of_samples_with_overlaps, number_of_short_edges

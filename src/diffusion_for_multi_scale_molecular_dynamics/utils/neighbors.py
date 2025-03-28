"""Neighbors.

The goal of this module is to compute list of neighbors within a given cutoff for
for positions in the unit cell of a periodic structure. The aim is to do this
efficiently on the GPU without CPU-GPU communications.
"""

from collections import namedtuple

import einops
import numpy as np
import torch
from pykeops.torch import LazyTensor

from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_positions_from_coordinates
from diffusion_for_multi_scale_molecular_dynamics.utils.lattice_utils import \
    get_relative_coordinates_lattice_vectors

INDEX_PADDING_VALUE = -1
POSITION_PADDING_VALUE = np.NaN


AdjacencyInfo = namedtuple(
    typename="AdjacencyInfo",
    field_names=[
        "adjacency_matrix",  # Adjacency matrix, in  COO format
        "shifts",  # lattice vector shifts when computing neighbor distances
        "edge_batch_indices",  # original batch index of each edge.
        "node_batch_indices",  # original batch index of each node.
        "number_of_edges",  # number of edges in each batch element.
    ],
)


def get_periodic_adjacency_information(
    cartesian_positions: torch.Tensor, basis_vectors: torch.Tensor, radial_cutoff: float, spatial_dimension: int = 3
) -> AdjacencyInfo:
    """Get periodic adjacency information.

    This method computes all the neighbors within the radial cutoff, accounting for periodicity, and returns
    various relevant data for building a geometric graph, such as all edges' source and target indices,
    the periodic shift between neighbors (ie, the lattice vector part of 'destination position minus
    origin position'). Because of periodicity, there may be multiple edges between the same two atoms.

    The algorithm assumes 3D space. The crystal structure is described by three non-collinear basis vectors,
    a1, a2 and a3, which induce a Bravais lattice. A 'relative coordinate' is given by [x1, x2, x3], where the
    real space position is "position = x1 a1 + x2 a2 + x3 a3". It is assumed that "0 <= x1, x2, x3 < 1".

    This method is meant to be applied to batches of structures and does not require the transfer
    of data to a different device (e.g., GPU to CPU). The input arrays have a batch dimension, but the
    output are flattened, as is typically assumed for batched graphs.

    N.B.:
        - The adjacency matrix outputs indices for the atoms in each structure, not global indices for the
          batch. These indices may have to be shifted when using this data structure to create a single
          disconnected graph that represents a batch of multiple connected graphs.

    Limitations:
        - It is assumed (but verified) that the radial cutoff is not so large that neighbors could be beyond the first
          shell of neighboring unit cells.
        - It is assumed that all structures have the same number of atoms.

    Args:
        cartesian_positions : atomic positions, assumed to be within the unit cell, in Euclidean space, in Angstrom.
                               Dimension [batch_size, max_number_of_atoms, spatial_dimension]
        basis_vectors : vectors that define the unit cell, (a1, a2, a3). The basis vectors are assumed
                        to be vertically stacked, namely
                                            [-- a1 --]
                                            [-- a2 --]
                                            [-- a3 --]
                        Dimension [batch_size, 3, 3].
        radial_cutoff : largest distance between neighbors, in Angstrom.
        spatial_dimension: the dimension of space.

    Returns:
        adjacency_info: an AdjacencyInfo object that contains
            * the adjacency matrix indices in the form of [source_indices,  destination_indices]
            * the lattice vector shifts between neighbors
            * the batch indices for each entry in the adjacency matrix
            * the batch indices for each node
            * the number of edges for each structure in the batch.
    """
    assert (
        len(cartesian_positions.shape) == 3
    ), "Wrong number of dimensions for relative_coordinates"
    assert len(basis_vectors.shape) == 3, "Wrong number of dimensions for basis_vectors"

    batch_size, max_natom, spatial_dimension_ = cartesian_positions.shape
    assert (
        spatial_dimension_ == spatial_dimension
    ), "Wrong spatial dimension for relative_coordinates "
    assert basis_vectors.shape == (
        batch_size,
        spatial_dimension,
        spatial_dimension,
    ), "Wrong shape for basis vectors"

    assert radial_cutoff > 0.0, "The radial cutoff should be greater than zero"

    device = cartesian_positions.device

    radial_cutoff = torch.tensor(radial_cutoff).to(device)
    zero = torch.tensor(0.0).to(device)

    # Check that the radial cutoff does not lead to possible neighbors beyond the first shell.
    shortest_cell_crossing_distances = _get_shortest_distance_that_crosses_unit_cell(
        basis_vectors, spatial_dimension=spatial_dimension
    )
    assert torch.all(shortest_cell_crossing_distances > radial_cutoff), (
        "The radial cutoff is so large that neighbors could be located "
        "beyond the first shell of periodic unit cell images."
    )

    # The relative coordinates lattice vectors have dimensions [number of lattice vectors, spatial_dimension]
    relative_lattice_vectors = get_relative_coordinates_lattice_vectors(
        number_of_shells=1, spatial_dimension=spatial_dimension
    ).to(device)
    number_of_relative_lattice_vectors = len(relative_lattice_vectors)

    # Repeat the relative lattice vectors along the batch dimension; the basis vectors could potentially be
    # different for every batch element.
    batched_relative_lattice_vectors = relative_lattice_vectors.repeat(batch_size, 1, 1)
    lattice_vectors = get_positions_from_coordinates(
        batched_relative_lattice_vectors, basis_vectors
    )

    # The shifted_positions are composed of the positions, which are located within the unit cell, shifted by
    # the various lattice vectors.
    #  Dimension [batch_size, number of relative lattice vectors, max_number_of_atoms, spatial_dimension].
    shifted_positions = _get_shifted_positions(cartesian_positions, lattice_vectors)

    # KeOps will be used to compute the distance matrix, |p_i - p_j |^2, without overflowing memory.
    #
    # In a nutshell, KeOps handle efficiently "kernels" of the form K_ij = f(x_i, x_j).
    # If  0 <= i < M and 0 <= j < N, the 'brute force' computation of K_ij would require O(MN) memory, which can
    # be very large. KeOps handles this kernel without storing it explicitly. The price to pay is that we have
    # to manipulate this abstraction of "virtual" dimensions, i and j.
    #
    # As stated in the KeOps documentation :
    #   https://www.kernel-operations.io/keops/_auto_tutorials/a_LazyTensors/plot_lazytensors_a.html
    #   """ Everything works just fine, with (...) major caveats:
    #
    #       - The structure of KeOps computations is still a little bit rigid: LazyTensors should only be used in
    #         situations where the large dimensions M and N over which the main reduction is performed are
    #         in positions -3 and -2 (respectively), with vector variables in position -1 and an arbitrary number of
    #         batch dimensions beforehand (...) """
    #
    # The positions arrays will have dimensions
    #       [batch_size, number of relative lattice vectors, max_natom, max_natom, spatial_dimension]
    #       |----- batch dimensions for KeOps -------------||--- i ---||--- j ---| |--- spatial ----|
    #
    # From the point of view of KeOps, the first 2 dimensions are "batch dimensions". The KeOps 'virtual' dimensions
    # are dim 2 and 3, which both corresponds to 'max_natom'.
    x_i = LazyTensor(
        cartesian_positions.view(batch_size, 1, max_natom, 1, spatial_dimension)
    )
    x_j = LazyTensor(
        shifted_positions.view(
            batch_size,
            number_of_relative_lattice_vectors,
            1,
            max_natom,
            spatial_dimension,
        )
    )

    # Symbolic matrix of squared distances
    #  Dimensions: [batch_size, number_of_relative_lattice_vectors, max_natom, max_natom]
    d_ij = ((x_i - x_j) ** 2).sum(dim=4)  # sum on the spatial_dimension variable.

    # Identify the number of neighbors within the cutoff distance for every atom.
    # This triggers a real computation, which involves a 'compilation' the first time around.
    # This compilation time is only paid once per code execution.
    max_k_array = (d_ij <= radial_cutoff**2).sum_reduction(
        dim=3
    )  # sum on "j", the second 'virtual' dimension.

    # This is the maximum number of neighbors for any atom in any structure in the batch.
    # Going forward, there is no need to look beyond this number of neighbors.
    max_number_of_neighbors = int(max_k_array.max())

    # Use KeOps KNN functionalities to find neighbors and their indices.
    squared_distances, dst_indices = d_ij.Kmin_argKmin(
        K=max_number_of_neighbors, dim=3
    )  # find neighbors along "j"
    # Dimensions: [batch_size, number_of_relative_lattice_vectors, max_natom, max_number_of_neighbors]
    # The 'dst_indices' array corresponds to KeOps first 'virtual' dimension (the "i" dimension). This goes from
    # 0 to max_atom - 1 and correspond to atom indices (specifically, destination indices!).

    # Identify neighbors within the radial_cutoff, but avoiding self.
    valid_neighbor_mask = torch.logical_and(
        zero < squared_distances, squared_distances <= radial_cutoff**2
    )
    # Dimensions: [batch_size, number_of_relative_lattice_vectors, max_natom, max_number_of_neighbors]

    # Combine all the non-batch dimensions to obtain the maximum number of edges per batch element
    number_of_edges = valid_neighbor_mask.view(batch_size, -1).sum(dim=1)

    # Each positive entry in the valid_neighbor_mask boolean array represents a valid edge. The indices of
    # this positive entry are also the indices of the corresponding edge components.
    valid_indices = valid_neighbor_mask.nonzero()
    # Dimension : [total number of edges, 4].
    # The columns correspond to (batch index, lattice vector, source atom, neighbor index)

    # Extract all the relevant components
    destination_indices = dst_indices[valid_neighbor_mask]
    source_indices = valid_indices[:, 2]
    edge_batch_indices = valid_indices[:, 0]
    lattice_vector_indices = valid_indices[:, 1]

    adjacency_matrix_coo_format = torch.stack([source_indices, destination_indices])

    lattice_vector_shifts = _get_vectors_from_multiple_indices(
        lattice_vectors, edge_batch_indices, lattice_vector_indices
    )

    return AdjacencyInfo(
        adjacency_matrix=adjacency_matrix_coo_format,
        shifts=lattice_vector_shifts,
        edge_batch_indices=edge_batch_indices,
        node_batch_indices=torch.repeat_interleave(torch.arange(batch_size), max_natom),
        number_of_edges=number_of_edges,
    )


def _get_shifted_positions(
    cartesian_positions: torch.Tensor, lattice_vectors: torch.Tensor
) -> torch.Tensor:
    """Get shifted positions.

    Args:
        cartesian_positions : atomic positions within the unit cell.
            Dimension [batch_size, max_number_of_atoms, 3].
        lattice_vectors : Bravais lattice vectors connecting the unit cell to its neighbors (and to itself!).
            Dimension [batch_size, number of relative lattice vectors, 3].

    Returns:
        shifted_positions:  the positions within the unit cell, shifted by the lattice vectors.
            Dimension [batch_size, number of relative lattice vectors, max_number_of_atoms, 3].
    """
    shifted_positions = (
        cartesian_positions[:, None, :, :] + lattice_vectors[:, :, None, :]
    )
    return shifted_positions


def _get_shortest_distance_that_crosses_unit_cell(
    basis_vectors: torch.Tensor, spatial_dimension: int = 3
) -> torch.Tensor:
    """Get the shortest distance that crosses unit cell.

    This method computes the shortest distance that crosses the unit cell.

    e.g.,            ---------------------------
                    /    ^                    /
                   /     |                   /
                  /      d                  /
                 /       |                 /
                /        v                /
               ---------------------------

    Args:
        basis_vectors : basis vectors that define the unit cell.
                        Dimension [batch_size, spatial_dimension]

    Returns:
        shortest_distances: shortest distance that can cross the unit cell, from one side to its other parallel side.
                            Dimension [batch_size].
    """
    assert spatial_dimension in {1, 2, 3}, "The spatial dimension must be 1, 2 or 3."
    assert len(basis_vectors.shape) == 3, "basis_vectors has wrong shape."
    assert (
        basis_vectors.shape[1] == spatial_dimension
    ), "Basis vectors in wrong spatial dimension."
    assert (
        basis_vectors.shape[2] == spatial_dimension
    ), "Basis vectors in wrong spatial dimension."

    match spatial_dimension:
        case 1:
            return _get_shortest_distance_that_crosses_unit_cell_1d(basis_vectors)
        case 2:
            return _get_shortest_distance_that_crosses_unit_cell_2d(basis_vectors)
        case 3:
            return _get_shortest_distance_that_crosses_unit_cell_3d(basis_vectors)
        case _:
            raise RuntimeError("Spatial dimension must be 1, 2 or 3.")


def _get_shortest_distance_that_crosses_unit_cell_1d(
    basis_vectors: torch.Tensor,
) -> torch.Tensor:
    """Get the shortest distance that crosses unit cell in 1D."""
    distances = basis_vectors.norm(dim=[-1, -2])
    return distances


def _get_shortest_distance_that_crosses_unit_cell_2d(
    basis_vectors: torch.Tensor,
) -> torch.Tensor:
    """Get the shortest distance that crosses unit cell in 2D."""
    a1 = basis_vectors[:, 0, :]
    a2 = basis_vectors[:, 1, :]

    dot_product = einops.einsum(a1, a2, "b i, b i -> b")

    norm_a1 = torch.norm(a1, dim=-1)
    norm_a2 = torch.norm(a2, dim=-1)

    orthogonal_a2 = a2 - (dot_product / norm_a1**2).unsqueeze(1) * a1
    distances_1 = orthogonal_a2.norm(dim=-1)

    orthogonal_a1 = a1 - (dot_product / norm_a2**2).unsqueeze(1) * a2
    distances_2 = orthogonal_a1.norm(dim=-1)

    distances = (
        torch.stack([distances_1, distances_2], dim=1).min(dim=1).values
    )

    return distances


def _get_shortest_distance_that_crosses_unit_cell_3d(
    basis_vectors: torch.Tensor,
) -> torch.Tensor:
    """Get the shortest distance that crosses unit cell in 3D."""
    # It is straightforward to show that the distance between two parallel planes,
    # (say the plane spanned by (a1, a2) crossing the origin and the plane spanned by (a1, a2) crossing the point a3)
    # is given by unit_normal DOT a3. The unit normal to the plane is proportional to the cross product of a1 and a2.
    #
    # This idea must be repeated for the three pairs of planes bounding the unit cell.
    a1 = basis_vectors[:, 0, :]
    a2 = basis_vectors[:, 1, :]
    a3 = basis_vectors[:, 2, :]

    cross_product_12 = torch.linalg.cross(a1, a2, dim=1)
    cross_product_13 = torch.linalg.cross(a1, a3, dim=1)
    cross_product_23 = torch.linalg.cross(a2, a3, dim=1)

    cell_volume = torch.abs((cross_product_12 * a3).sum(dim=1))

    distances_12 = cell_volume / torch.linalg.norm(cross_product_12, dim=1)
    distances_13 = cell_volume / torch.linalg.norm(cross_product_13, dim=1)
    distances_23 = cell_volume / torch.linalg.norm(cross_product_23, dim=1)

    distances = (
        torch.stack([distances_12, distances_13, distances_23], dim=1).min(dim=1).values
    )

    return distances


def _get_vectors_from_multiple_indices(
    vectors: torch.Tensor, batch_indices: torch.Tensor, vector_indices: torch.Tensor
) -> torch.Tensor:
    """Get vectors from multiple indices.

    Extract vectors in higher dimensional tensor from multiple indices array.

    This functionality is sufficiently arcane in pytorch to warrant its own function that we can
    test robustly.

    Args:
        vectors : Tensor with multiple 'batch-like' indices that contain vectors.
                    Dimensions : [batch_size, number_of_vectors, spatial_dimension].
        batch_indices : Array of indices for the batch dimension. All values should be between 0 and batch_size - 1.
                    Dimensions : [number_of_indices]
        vector_indices : Array of indices for the 'number of vectors' dimension. All values should be between
                        0 and number_of_vectors - 1.
                    Dimensions : [number_of_indices]

    Returns:
        indexed_vectors: the vectors properly indexed by batch_indices and vector_indices.
                         Dimensions : [number_of_indices, spatial_dimension]
    """
    # Yes, the answer looks trivial, but it's quite confusing to know if if gather or select_index should be used!
    return vectors[batch_indices, vector_indices]


def shift_adjacency_matrix_indices_for_graph_batching(
    adjacency_matrix: torch.Tensor, num_edges: torch.Tensor, number_of_atoms: int
) -> torch.Tensor:
    """Shift the node indices in the adjacency matrix for graph batching."""
    index_shifts = (
        torch.arange(len(num_edges)).to(adjacency_matrix.device) * number_of_atoms
    )
    adj_shifts = torch.repeat_interleave(index_shifts, num_edges).repeat(2, 1)

    return adjacency_matrix + adj_shifts

"""Neighbors.

The goal of this module is to compute list of neighbors within a given cutoff for
for positions in the unit cell of a periodic structure. The aim is to do this
efficiently on the GPU without CPU-GPU communications.
"""
import itertools
from typing import Tuple

import numpy as np
import torch
from pykeops.torch import LazyTensor

INDEX_PADDING_VALUE = -1
POSITION_PADDING_VALUE = np.NaN


def get_periodic_neighbor_indices_and_displacements(relative_coordinates: torch.Tensor,
                                                    basis_vectors: torch.Tensor,
                                                    radial_cutoff: float) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get periodic neighbor indices and displacements.

    This method computes all the neighbors within the radial cutoff, accounting for periodicity, and returns
    the edges' source and target indices as well as the corresponding displacements (ie, destination position minus
    origin position). Because of periodicity, there may be multiple edges between the same two atoms.

    The algorithm assumes 3D space. The crystal structure is described by three non-collinear basis vectors,
    a1, a2 and a3, which induce a Bravais lattice. A 'relative coordinate' is given by [x1, x2, x3], where the
    real space position is "position = x1 a1 + x2 a2 + x3 a3". It is assumed that "0 <= x1, x2, x3 < 1".

    This method is meant to be applied to batches of structures and does not require the transfer
    of data to a different device (e.g., GPU to CPU).

    Limitations:
        - It is assumed that the radial cutoff is not so large that neighbors could be beyond the first
          shell of neighboring unit cells.
        - It is assumed that all structures have the same number of atoms.


    Args:
        relative_coordinates : atomic coordinates within the unit cell.
                               Dimension [batch_size, max_number_of_atoms, 3]
        basis_vectors : vectors that define the unit cell, (a1, a2, a3). The basis vectors are assumed
                        to be vertically stacked, namely
                                            [-- a1 --]
                                            [-- a2 --]
                                            [-- a3 --]
                        Dimension [batch_size, 3, 3].
        radial_cutoff : largest distance between neighbors.

    Returns:
        source_indices : indices of the source positions. Dimension [batch_size, maximum_number_of_edges]
        destination_indices : indices of the destination positions. Dimension [batch_size, maximum_number_of_edges]
        displacements : displacement vector for each of the edges. Dimension [batch_size, maximum_number_of_edges, 3]
    """
    spatial_dimension = 3  # We define this to avoid "magic numbers" in the code below.
    assert len(relative_coordinates.shape) == spatial_dimension, "Wrong number of dimensions for relative_coordinates"
    assert len(basis_vectors.shape) == spatial_dimension, "Wrong number of dimensions for basis_vectors"

    batch_size, max_natom, spatial_dimension_ = relative_coordinates.shape
    assert spatial_dimension_ == spatial_dimension, "Wrong spatial dimension for relative_coordinates "
    assert basis_vectors.shape == (batch_size, spatial_dimension, spatial_dimension), "Wrong shape for basis vectors"

    assert radial_cutoff > 0., "The radial cutoff should be greater than zero"

    # TODO: check that the radial cutoff does not lead to possible neighbors beyond the first shell.

    device = relative_coordinates.device

    radial_cutoff = torch.tensor(radial_cutoff).to(device)
    zero = torch.tensor(0.0).to(device)

    shortest_cell_crossing_distances = _get_shortest_distance_that_crosses_unit_cell(basis_vectors)

    assert torch.all(shortest_cell_crossing_distances > radial_cutoff), \
        ("The radial cutoff is so large that neighbors could be located "
         "beyond the first shell of periodic unit cell images.")

    # The relative coordinates lattice vectors have dimensions [number of lattice vectors, spatial_dimension]
    relative_lattice_vectors = _get_relative_coordinates_lattice_vectors(number_of_shells=1).to(device)
    number_of_relative_lattice_vectors = len(relative_lattice_vectors)

    # The shifted_relative_coordinates are composed of the relative coordinates, which are located
    # within the unit cell, shifted by the various relative lattice vectors.
    #   Dimension [batch_size, max_number_of_atoms, number_of_relative_lattice_vectors, spatial_dimension].
    shifted_relative_coordinates = _get_shifted_relative_coordinates(relative_coordinates, relative_lattice_vectors)

    # Compute the positions in Euclidean space.
    #   The positions are defined as p = c1 a1 + c2 a2 + c3 a3, which can be expressed as
    #   (p_x, p_y, p_z) = [c1, c2, c3] [  - a1 - ]
    #                                  [  - a2 - ]
    #                                  [  - a3 - ]
    positions = torch.matmul(relative_coordinates, basis_vectors)
    shifted_positions = torch.matmul(shifted_relative_coordinates, basis_vectors[:, None, :, :])

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
    x_i = LazyTensor(positions.view(batch_size, 1, max_natom, 1, spatial_dimension))
    x_j = LazyTensor(shifted_positions.view(batch_size, number_of_relative_lattice_vectors, 1,
                                            max_natom, spatial_dimension))

    # Symbolic matrix of squared distances
    #  Dimensions: [batch_size, number_of_relative_lattice_vectors, max_natom, max_natom]
    d_ij = ((x_i - x_j) ** 2).sum(dim=4)  # sum on the spatial_dimension variable.

    # Identify the number of neighbors within the cutoff distance for every atom.
    # This triggers a real computation, which involves a 'compilation' the first time around.
    # This compilation time is only paid once per code execution.
    max_k_array = (d_ij <= radial_cutoff**2).sum_reduction(dim=3)  # sum on "j", the second 'virtual' dimension.

    # This is the maximum number of neighbors for any atom in any structure in the batch.
    # Going forward, there is no need to look beyond this number of neighbors.
    max_number_of_neighbors = int(max_k_array.max())

    # Use KeOps KNN functionalities to find neighbors and their indices.
    squared_distances, dst_indices = d_ij.Kmin_argKmin(K=max_number_of_neighbors, dim=3)  # find neighbors along "j"
    # Dimensions: [batch_size, number_of_relative_lattice_vectors, max_natom, max_number_of_neighbors]
    # The 'dst_indices' array corresponds to KeOps first 'virtual' dimension (the "i" dimension). This goes from
    # 0 to max_atom - 1 and correspond to atom indices (specifically, destination indices!).

    # Build the source indices to have the same dimensions as dst_indices.
    shape = (batch_size, number_of_relative_lattice_vectors, max_number_of_neighbors, 1)
    src_indices = torch.arange(max_natom).repeat(shape).transpose(-2, -1)
    # src_indices  is constant along all dimensions except the "i" dimension, where it goes from 0 to max_atom - 1.
    # Dimensions: [batch_size, number_of_relative_lattice_vectors, max_natom, max_number_of_neighbors]

    # Identify neighbors within the radial_cutoff, but avoiding self.
    valid_neighbor_mask = torch.logical_and(zero < squared_distances, squared_distances <= radial_cutoff**2)
    # Dimensions: [batch_size, number_of_relative_lattice_vectors, max_natom, max_number_of_neighbors]

    # Combine all the non-batch dimensions.
    flattened_mask = valid_neighbor_mask.view(batch_size, -1)
    max_edge_per_structure_in_batch = flattened_mask.sum(dim=1)

    # Broadcast the source and destination positions to the same dimensions so that they can be
    # treated in the same way.
    origins = torch.take_along_dim(positions[:, None, :, None, :], src_indices[:, :, :, :, None], dim=2)
    destinations = torch.take_along_dim(shifted_positions[:, :, :, None, :], dst_indices[:, :, :, :, None], dim=2)
    # Dimension: [batch_size, number of lattice vectors, max_natom, max number of neighbors, spatial_dimension]

    # Contract non-batch dimensions.
    flat_src_indices = src_indices.reshape(batch_size, -1)[flattened_mask]
    flat_dst_indices = dst_indices.reshape(batch_size, -1)[flattened_mask]
    flat_origins = origins.reshape(batch_size, -1, spatial_dimension)[flattened_mask]
    flat_destinations = destinations.reshape(batch_size, -1, spatial_dimension)[flattened_mask]
    # The indices arrays have dimensions [total number of edges in batch], and the
    # vector arrays have dimensions [total number of edges in batch, 3].

    # Only compute what is needed!
    flat_displacements = flat_destinations - flat_origins

    # Chop back into one item per batch entry, and then pad.
    split_points = torch.cumsum(max_edge_per_structure_in_batch, dim=0)[:-1]

    list_src_indices = torch.tensor_split(flat_src_indices, split_points)
    source_indices = torch.nn.utils.rnn.pad_sequence(list_src_indices,
                                                     batch_first=True,
                                                     padding_value=INDEX_PADDING_VALUE)

    list_dst_indices = torch.tensor_split(flat_dst_indices, split_points)
    destination_indices = torch.nn.utils.rnn.pad_sequence(list_dst_indices,
                                                          batch_first=True,
                                                          padding_value=INDEX_PADDING_VALUE)

    list_displacements = torch.tensor_split(flat_displacements, split_points)
    displacements = torch.nn.utils.rnn.pad_sequence(list_displacements,
                                                    batch_first=True,
                                                    padding_value=POSITION_PADDING_VALUE)

    return source_indices, destination_indices, displacements


def _get_relative_coordinates_lattice_vectors(number_of_shells: int = 1) -> torch.Tensor:
    """Get relative coordinates lattice vectors.

    Get all the lattice vectors in relative coordinates from -number_of_shells to +number_of_shells,
    in every spatial directions.

    Args:
        number_of_shells: number of shifts along lattice vectors in the positive direction.

    Returns:
        list_relative_lattice_vectors : all the lattice vectors in relative coordinates (ie, integers).
    """
    spatial_dimension = 3
    shifts = range(-number_of_shells, number_of_shells + 1)
    list_relative_lattice_vectors = torch.tensor(list(itertools.product(shifts, repeat=spatial_dimension)))
    return list_relative_lattice_vectors


def _get_shifted_relative_coordinates(relative_coordinates: torch.Tensor,
                                      relative_lattice_vectors: torch.Tensor) -> torch.Tensor:
    """Get shifted relative coordinates.

    Args:
        relative_coordinates : relative coordinates within the unit cell.
            Dimension [batch_size, max_number_of_atoms, 3].
        relative_lattice_vectors : relative coordinates lattice vectors (ie, array of integer triplets).
            Dimension [number of relative lattice vectors, 3].

    Returns:
        shifted_relative_coordinates: the unit cell relative coordinates, shifted by the lattice vectors.
            Dimension [batch_size, number of relative lattice vectors, max_number_of_atoms, 3].
    """
    shifted_coordinates = relative_coordinates[:, :, None, :] + relative_lattice_vectors[None, None, :, :]
    shifted_coordinates = shifted_coordinates.transpose(2, 1)
    return shifted_coordinates


def _get_shortest_distance_that_crosses_unit_cell(basis_vectors: torch.Tensor) -> torch.Tensor:
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
                        Dimension [batch_size, spatial_dimension = 3]

    Returns:
        shortest_distances: shortest distance that can cross the unit cell, from one side to its other parallel side.
                            Dimension [batch_size].
    """
    # It is straightforward to show that the distance between two parallel planes,
    # (say the plane spanned by (a1, a2) crossing the origin and the plane spanned by (a1, a2) crossing the point a3)
    # is given by unit_normal DOT a3. The unit normal to the plane is proportional to the cross product of a1 and a2.
    #
    # This idea must be repeated for the three pairs of planes bounding the unit cell.
    spatial_dimension = 3
    assert len(basis_vectors.shape) == 3, "basis_vectors has wrong shape."
    assert basis_vectors.shape[1] == spatial_dimension, "Basis vectors in wrong spatial dimension."
    assert basis_vectors.shape[2] == spatial_dimension, "Basis vectors in wrong spatial dimension."
    a1 = basis_vectors[:, 0, :]
    a2 = basis_vectors[:, 1, :]
    a3 = basis_vectors[:, 2, :]

    cross_product_12 = torch.cross(a1, a2)
    cross_product_13 = torch.cross(a1, a3)
    cross_product_23 = torch.cross(a2, a3)

    cell_volume = torch.abs((cross_product_12 * a3).sum(dim=1))

    distances_12 = cell_volume / torch.linalg.norm(cross_product_12, dim=1)
    distances_13 = cell_volume / torch.linalg.norm(cross_product_13, dim=1)
    distances_23 = cell_volume / torch.linalg.norm(cross_product_23, dim=1)

    distances = torch.stack([distances_12, distances_13, distances_23], dim=1).min(dim=1).values

    return distances

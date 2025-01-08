from collections import namedtuple

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_positions_from_coordinates
from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import (
    AdjacencyInfo, _get_relative_coordinates_lattice_vectors,
    _get_shifted_positions, _get_shortest_distance_that_crosses_unit_cell,
    _get_vectors_from_multiple_indices, get_periodic_adjacency_information,
    shift_adjacency_matrix_indices_for_graph_batching)
from tests.fake_data_utils import find_aligning_permutation

Neighbors = namedtuple(
    "Neighbors", ["source_index", "destination_index", "displacement", "shift"]
)


@pytest.fixture(scope="module", autouse=True)
def set_seed():
    """Set the random seed."""
    torch.manual_seed(2342323)


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def number_of_atoms():
    return 32


@pytest.fixture()
def spatial_dimension(request):
    return 3


@pytest.fixture
def basis_vectors(batch_size, spatial_dimension):
    # orthogonal boxes with dimensions between 5 and 10.
    orthogonal_boxes = torch.stack(
        [torch.diag(5.0 + 5.0 * torch.rand(spatial_dimension)) for _ in range(batch_size)]
    )
    # add a bit of noise to make the vectors not quite orthogonal
    basis_vectors = orthogonal_boxes + 0.1 * torch.randn(batch_size, spatial_dimension, spatial_dimension)
    return basis_vectors


@pytest.fixture
def relative_coordinates(batch_size, number_of_atoms, spatial_dimension):
    return torch.rand(batch_size, number_of_atoms, spatial_dimension)


@pytest.fixture
def positions(relative_coordinates, basis_vectors):
    positions = get_positions_from_coordinates(relative_coordinates, basis_vectors)
    return positions


@pytest.fixture
def lattice_vectors(batch_size, basis_vectors, number_of_shells, spatial_dimension):
    relative_lattice_vectors = _get_relative_coordinates_lattice_vectors(
        number_of_shells, spatial_dimension
    )
    batched_relative_lattice_vectors = relative_lattice_vectors.repeat(batch_size, 1, 1)
    lattice_vectors = get_positions_from_coordinates(
        batched_relative_lattice_vectors, basis_vectors
    )

    return lattice_vectors


@pytest.fixture
def structures(basis_vectors, relative_coordinates):
    list_structures = []
    for basis, coordinates in zip(basis_vectors, relative_coordinates):
        number_of_atoms = coordinates.shape[0]
        species = number_of_atoms * [
            "Si"
        ]  # this is a dummy variable. It doesn't matter what the atom types are...
        lattice = Lattice(matrix=basis.cpu().numpy(), pbc=(True, True, True))

        structure = Structure(
            lattice=lattice,
            species=species,
            coords=coordinates.cpu().numpy(),
            to_unit_cell=False,  # already in [0, 1) range.
            coords_are_cartesian=False,
        )
        list_structures.append(structure)

    return list_structures


@pytest.fixture
def expected_neighbors(structures, radial_cutoff):
    """Build the list of neighbors for each structure using Pymatgen."""
    structure_neighbors = []
    for structure in structures:
        list_neighbor_sites = structure.get_all_neighbors(r=radial_cutoff)

        list_neighbors = []
        for src_idx, neighbor_sites in enumerate(list_neighbor_sites):
            source_site = structure.sites[src_idx]
            for neighbor_site in neighbor_sites:
                # Make sure the neighbors are no more than one shell away.
                assert np.all(
                    np.abs(np.array(neighbor_site.image)) <= 1.0
                ), "A neighbor site is out of range. Review test code!"

                dst_idx = neighbor_site.index
                displacement = neighbor_site.coords - source_site.coords

                shift = np.dot(np.array(neighbor_site.image), structure.lattice.matrix)

                neighbors = Neighbors(
                    source_index=src_idx,
                    destination_index=dst_idx,
                    displacement=displacement,
                    shift=shift,
                )
                list_neighbors.append(neighbors)

        structure_neighbors.append(list_neighbors)

    return structure_neighbors


@pytest.fixture
def expected_adjacency_info(structures, expected_neighbors):
    node_batch_indices = []
    edge_batch_indices = []
    shifts = []
    adj_matrix = []
    number_of_edges = []
    for batch_index, (structure, list_neighbors) in enumerate(
        zip(structures, expected_neighbors)
    ):
        number_of_atoms = len(structure)
        batch_node_batch_indices = torch.tensor(
            number_of_atoms * [batch_index], dtype=torch.long
        )
        node_batch_indices.append(batch_node_batch_indices)

        batch_shifts = torch.stack(
            [torch.from_numpy(neigh.shift) for neigh in list_neighbors]
        )
        shifts.append(batch_shifts)

        source_indices = torch.tensor([neigh.source_index for neigh in list_neighbors])
        dest_indices = torch.tensor(
            [neigh.destination_index for neigh in list_neighbors]
        )
        batch_adj_matrix = torch.stack([source_indices, dest_indices])
        adj_matrix.append(batch_adj_matrix)

        number_of_edges.append(len(source_indices))

        batch_edge_batch_indices = torch.tensor(
            len(source_indices) * [batch_index], dtype=torch.long
        )
        edge_batch_indices.append(batch_edge_batch_indices)

    return AdjacencyInfo(
        adjacency_matrix=torch.cat(adj_matrix, dim=1),
        shifts=torch.cat(shifts, dim=0),
        node_batch_indices=torch.cat(node_batch_indices),
        edge_batch_indices=torch.cat(edge_batch_indices),
        number_of_edges=torch.tensor(number_of_edges),
    )


@pytest.mark.parametrize("radial_cutoff", [1.1, 2.2, 3.3])
def test_get_periodic_adjacency_information(
    basis_vectors, positions, radial_cutoff, batch_size, expected_adjacency_info
):
    computed_adjacency_info = get_periodic_adjacency_information(
        positions, basis_vectors, radial_cutoff
    )

    torch.testing.assert_close(
        expected_adjacency_info.number_of_edges, computed_adjacency_info.number_of_edges
    )
    torch.testing.assert_close(
        expected_adjacency_info.node_batch_indices,
        computed_adjacency_info.node_batch_indices,
    )
    torch.testing.assert_close(
        expected_adjacency_info.edge_batch_indices,
        computed_adjacency_info.edge_batch_indices,
    )

    # The edges might not be in the same order; build the corresponding permutation.
    for batch_idx in range(batch_size):
        expected_mask = expected_adjacency_info.edge_batch_indices == batch_idx
        expected_src_idx, expected_dst_idx = expected_adjacency_info.adjacency_matrix[
            :, expected_mask
        ]
        expected_shifts = expected_adjacency_info.shifts[expected_mask, :]
        expected_displacements = (
            positions[batch_idx, expected_dst_idx]
            + expected_shifts
            - positions[batch_idx, expected_src_idx]
        )

        computed_mask = computed_adjacency_info.edge_batch_indices == batch_idx
        computed_src_idx, computed_dst_idx = computed_adjacency_info.adjacency_matrix[
            :, computed_mask
        ]
        computed_shifts = computed_adjacency_info.shifts[computed_mask, :]
        computed_displacements = (
            positions[batch_idx, computed_dst_idx]
            + computed_shifts
            - positions[batch_idx, computed_src_idx]
        )

        permutation_indices = find_aligning_permutation(
            expected_displacements, computed_displacements, tol=1e-5
        )

        torch.testing.assert_close(
            computed_shifts[permutation_indices], expected_shifts, check_dtype=False
        )
        torch.testing.assert_close(
            computed_src_idx[permutation_indices], expected_src_idx
        )
        torch.testing.assert_close(
            computed_dst_idx[permutation_indices], expected_dst_idx
        )


@pytest.mark.parametrize("spatial_dimension", [1, 2, 3])
def test_get_periodic_neighbour_indices_and_displacements_large_cutoff(
    basis_vectors, relative_coordinates, spatial_dimension
):
    # Check that the code crashes if the radial cutoff is too big!
    shortest_cell_crossing_distances = _get_shortest_distance_that_crosses_unit_cell(
        basis_vectors, spatial_dimension=spatial_dimension
    ).min()

    large_radial_cutoff = (shortest_cell_crossing_distances + 0.1).item()
    small_radial_cutoff = (shortest_cell_crossing_distances - 0.1).item()

    # Should run
    get_periodic_adjacency_information(
        relative_coordinates, basis_vectors, small_radial_cutoff, spatial_dimension=spatial_dimension
    )

    with pytest.raises(AssertionError):
        # Should crash
        get_periodic_adjacency_information(
            relative_coordinates, basis_vectors, large_radial_cutoff, spatial_dimension=spatial_dimension
        )


@pytest.mark.parametrize("number_of_shells", [1, 2, 3])
def test_get_relative_coordinates_lattice_vectors_1d(number_of_shells):

    expected_lattice_vectors = []

    for nx in torch.arange(-number_of_shells, number_of_shells + 1):
        lattice_vector = torch.tensor([nx])
        expected_lattice_vectors.append(lattice_vector)

    expected_lattice_vectors = torch.stack(expected_lattice_vectors).to(
        dtype=torch.float32
    )
    computed_lattice_vectors = _get_relative_coordinates_lattice_vectors(
        number_of_shells, spatial_dimension=1
    )

    torch.testing.assert_close(expected_lattice_vectors, computed_lattice_vectors)


@pytest.mark.parametrize("number_of_shells", [1, 2, 3])
def test_get_relative_coordinates_lattice_vectors_2d(number_of_shells):

    expected_lattice_vectors = []

    for nx in torch.arange(-number_of_shells, number_of_shells + 1):
        for ny in torch.arange(-number_of_shells, number_of_shells + 1):
            lattice_vector = torch.tensor([nx, ny])
            expected_lattice_vectors.append(lattice_vector)

    expected_lattice_vectors = torch.stack(expected_lattice_vectors).to(
        dtype=torch.float32
    )
    computed_lattice_vectors = _get_relative_coordinates_lattice_vectors(
        number_of_shells, spatial_dimension=2
    )

    torch.testing.assert_close(expected_lattice_vectors, computed_lattice_vectors)


@pytest.mark.parametrize("number_of_shells", [1, 2, 3])
def test_get_relative_coordinates_lattice_vectors_3d(number_of_shells):

    expected_lattice_vectors = []

    for nx in torch.arange(-number_of_shells, number_of_shells + 1):
        for ny in torch.arange(-number_of_shells, number_of_shells + 1):
            for nz in torch.arange(-number_of_shells, number_of_shells + 1):
                lattice_vector = torch.tensor([nx, ny, nz])
                expected_lattice_vectors.append(lattice_vector)

    expected_lattice_vectors = torch.stack(expected_lattice_vectors).to(
        dtype=torch.float32
    )
    computed_lattice_vectors = _get_relative_coordinates_lattice_vectors(
        number_of_shells, spatial_dimension=3
    )

    torch.testing.assert_close(expected_lattice_vectors, computed_lattice_vectors)


@pytest.mark.parametrize("number_of_shells", [1, 2])
def test_get_shifted_positions(positions, lattice_vectors):

    computed_shifted_positions = _get_shifted_positions(positions, lattice_vectors)

    batch_size, number_of_atoms, spatial_dimension = positions.shape
    batch_size_, number_of_lattice_vectors, spatial_dimension_ = lattice_vectors.shape

    assert batch_size == batch_size_
    assert spatial_dimension == spatial_dimension_ == 3

    for batch_idx in range(batch_size):
        for atom_idx in range(number_of_atoms):
            position = positions[batch_idx, atom_idx, :]
            for lat_idx in range(number_of_lattice_vectors):
                lattice_vector = lattice_vectors[batch_idx, lat_idx, :]

                expected_cell_shift_positions = position + lattice_vector
                computed_cell_shift_positions = computed_shifted_positions[
                    batch_idx, lat_idx, atom_idx, :
                ]

                torch.testing.assert_close(
                    expected_cell_shift_positions, computed_cell_shift_positions
                )


def test_get_shortest_distance_that_crosses_unit_cell_3d(basis_vectors):
    expected_shortest_distances = []
    for matrix in basis_vectors.numpy():
        a1, a2, a3 = matrix

        cross_product_12 = np.cross(a1, a2)
        cross_product_13 = np.cross(a1, a3)
        cross_product_23 = np.cross(a2, a3)

        d1 = np.abs(np.dot(cross_product_23, a1)) / np.linalg.norm(cross_product_23)
        d2 = np.abs(np.dot(cross_product_13, a2)) / np.linalg.norm(cross_product_13)
        d3 = np.abs(np.dot(cross_product_12, a3)) / np.linalg.norm(cross_product_12)

        expected_shortest_distances.append(np.min([d1, d2, d3]))

    expected_shortest_distances = torch.Tensor(expected_shortest_distances)
    computed_shortest_distances = _get_shortest_distance_that_crosses_unit_cell(
        basis_vectors
    )

    torch.testing.assert_close(expected_shortest_distances, computed_shortest_distances)


def test_get_vectors_from_multiple_indices(batch_size, number_of_atoms):

    spatial_dimension = 3
    vectors = torch.rand(batch_size, number_of_atoms, spatial_dimension)

    number_of_indices = 100

    batch_indices = torch.randint(0, batch_size, (number_of_indices,))
    vector_indices = torch.randint(0, number_of_atoms, (number_of_indices,))

    expected_vectors = torch.empty(
        number_of_indices, spatial_dimension, dtype=torch.float32
    )

    for idx, (batch_idx, vector_idx) in enumerate(zip(batch_indices, vector_indices)):
        expected_vectors[idx] = vectors[batch_idx, vector_idx]

    computed_vectors = _get_vectors_from_multiple_indices(
        vectors, batch_indices, vector_indices
    )

    torch.testing.assert_close(expected_vectors, computed_vectors)


def test_shift_adjacency_matrix_indices_for_graph_batching(batch_size, number_of_atoms):

    num_edges = torch.randint(10, 100, (batch_size,))
    total_number_of_edges = num_edges.sum()
    adjacency_matrix = torch.randint(number_of_atoms, (2, total_number_of_edges))
    expected_shifted_adj = torch.clone(adjacency_matrix)

    cumulative_edges = torch.cumsum(num_edges, dim=0)
    index_shift = number_of_atoms
    for bottom_idx, top_idx in zip(cumulative_edges[:-1], cumulative_edges[1:]):
        expected_shifted_adj[:, bottom_idx:top_idx] += index_shift
        index_shift += number_of_atoms

    computed_shifted_adj = shift_adjacency_matrix_indices_for_graph_batching(
        adjacency_matrix, num_edges, number_of_atoms
    )

    torch.testing.assert_close(expected_shifted_adj, computed_shifted_adj)

from collections import namedtuple

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from crystal_diffusion.utils.neighbors import (
    INDEX_PADDING_VALUE, POSITION_PADDING_VALUE,
    _get_relative_coordinates_lattice_vectors,
    _get_shifted_relative_coordinates,
    _get_shortest_distance_that_crosses_unit_cell,
    get_periodic_neighbor_indices_and_displacements)
from tests.fake_data_utils import find_aligning_permutation

Neighbors = namedtuple("Neighbors", ["source_index", "destination_index", "displacement"])


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


@pytest.fixture
def basis_vectors(batch_size):
    # orthogonal boxes with dimensions between 5 and 10.
    orthogonal_boxes = torch.stack([torch.diag(5. + 5. * torch.rand(3)) for _ in range(batch_size)])
    # add a bit of noise to make the vectors not quite orthogonal
    basis_vectors = orthogonal_boxes + 0.1 * torch.randn(batch_size, 3, 3)
    return basis_vectors


@pytest.fixture
def relative_coordinates(batch_size, number_of_atoms):
    return torch.rand(batch_size, number_of_atoms, 3)


@pytest.fixture
def structures(basis_vectors, relative_coordinates):
    list_structures = []
    for basis, coordinates in zip(basis_vectors, relative_coordinates):
        number_of_atoms = coordinates.shape[0]
        species = number_of_atoms * ["Si"]  # this is a dummy variable. It doesn't matter what the atom types are...
        # TODO
        lattice = Lattice(matrix=basis.cpu().numpy(), pbc=(True, True, True))

        structure = Structure(lattice=lattice,
                              species=species,
                              coords=coordinates.cpu().numpy(),
                              to_unit_cell=False,  # already in [0, 1) range.
                              coords_are_cartesian=False)
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
                assert np.all(np.abs(np.array(neighbor_site.image)) <= 1.), \
                    "A neighbor site is out of range. Review test code!"

                dst_idx = neighbor_site.index
                displacement = neighbor_site.coords - source_site.coords

                neighbors = Neighbors(source_index=src_idx,
                                      destination_index=dst_idx,
                                      displacement=displacement)
                list_neighbors.append(neighbors)

        structure_neighbors.append(list_neighbors)

    return structure_neighbors


@pytest.fixture
def expected_neighbor_indices_and_displacements(expected_neighbors):

    list_source_indices = []
    list_dest_indices = []
    list_displacements = []
    for list_neighbors in expected_neighbors:
        source_indices = torch.tensor([neigh.source_index for neigh in list_neighbors])
        list_source_indices.append(source_indices)

        dest_indices = torch.tensor([neigh.destination_index for neigh in list_neighbors])
        list_dest_indices.append(dest_indices)
        displacements = torch.stack([torch.from_numpy(neigh.displacement) for neigh in list_neighbors])
        list_displacements.append(displacements)

    expected_source_indices = torch.nn.utils.rnn.pad_sequence(list_source_indices,
                                                              batch_first=True,
                                                              padding_value=INDEX_PADDING_VALUE)

    expected_destination_indices = torch.nn.utils.rnn.pad_sequence(list_dest_indices,
                                                                   batch_first=True,
                                                                   padding_value=INDEX_PADDING_VALUE)

    expected_displacements = torch.nn.utils.rnn.pad_sequence(list_displacements,
                                                             batch_first=True,
                                                             padding_value=POSITION_PADDING_VALUE)

    return expected_source_indices, expected_destination_indices, expected_displacements


# This test might be slow because KeOps needs to compile some stuff...
@pytest.mark.parametrize("radial_cutoff", [3.3])
def test_get_periodic_neighbour_indices_and_displacements(basis_vectors, relative_coordinates, radial_cutoff,
                                                          expected_neighbor_indices_and_displacements):
    expected_src_idx, expected_dst_idx, expected_displacements = expected_neighbor_indices_and_displacements

    computed_src_idx, computed_dst_idx, computed_displacements = (
        get_periodic_neighbor_indices_and_displacements(relative_coordinates, basis_vectors, radial_cutoff))

    batch_size, max_edges, spatial_dimension = expected_displacements.shape

    assert computed_src_idx.shape == (batch_size, max_edges)
    assert computed_dst_idx.shape == (batch_size, max_edges)
    assert computed_displacements.shape == (batch_size, max_edges, spatial_dimension)

    # Validate that the padding is the same in both computed and expected arrays
    torch.testing.assert_allclose(torch.isnan(computed_displacements), torch.isnan(expected_displacements))
    torch.testing.assert_allclose(expected_src_idx == INDEX_PADDING_VALUE, computed_src_idx == INDEX_PADDING_VALUE)
    torch.testing.assert_allclose(expected_dst_idx == INDEX_PADDING_VALUE, computed_dst_idx == INDEX_PADDING_VALUE)

    # The edges might not be in the same order. Check permutations
    for batch_idx in range(batch_size):

        batch_expected_src_idx = expected_src_idx[batch_idx]
        valid_count = (batch_expected_src_idx != INDEX_PADDING_VALUE).sum()

        batch_expected_src_idx = expected_src_idx[batch_idx][:valid_count]
        batch_computed_src_idx = computed_src_idx[batch_idx][:valid_count]

        batch_expected_dst_idx = expected_src_idx[batch_idx][:valid_count]
        batch_computed_dst_idx = computed_src_idx[batch_idx][:valid_count]

        batch_expected_displacements = expected_displacements[batch_idx][:valid_count]
        batch_computed_displacements = computed_displacements[batch_idx][:valid_count]

        permutation_indices = find_aligning_permutation(batch_expected_displacements,
                                                        batch_computed_displacements,
                                                        tol=1e-5)

        torch.testing.assert_allclose(batch_computed_displacements[permutation_indices], batch_expected_displacements)
        torch.testing.assert_allclose(batch_computed_src_idx[permutation_indices], batch_expected_src_idx)
        torch.testing.assert_allclose(batch_computed_dst_idx[permutation_indices], batch_expected_dst_idx)


def test_get_periodic_neighbour_indices_and_displacements_large_cutoff(basis_vectors, relative_coordinates):
    # Check that the code crashes if the radial cutoff is too big!
    shortest_cell_crossing_distances = _get_shortest_distance_that_crosses_unit_cell(basis_vectors).min()

    large_radial_cutoff = shortest_cell_crossing_distances + 0.1
    small_radial_cutoff = shortest_cell_crossing_distances - 0.1

    # Should run
    get_periodic_neighbor_indices_and_displacements(relative_coordinates, basis_vectors, small_radial_cutoff)

    with pytest.raises(AssertionError):
        # Should crash
        get_periodic_neighbor_indices_and_displacements(relative_coordinates, basis_vectors, large_radial_cutoff)


@pytest.mark.parametrize("number_of_shells", [1, 2, 3])
def test_get_relative_coordinates_lattice_vectors(number_of_shells):

    expected_lattice_vectors = []

    for nx in torch.arange(-number_of_shells, number_of_shells + 1):
        for ny in torch.arange(-number_of_shells, number_of_shells + 1):
            for nz in torch.arange(-number_of_shells, number_of_shells + 1):
                lattice_vector = torch.tensor([nx, ny, nz])
                expected_lattice_vectors.append(lattice_vector)

    expected_lattice_vectors = torch.stack(expected_lattice_vectors)
    computed_lattice_vectors = _get_relative_coordinates_lattice_vectors(number_of_shells)

    torch.testing.assert_allclose(expected_lattice_vectors, computed_lattice_vectors)


@pytest.mark.parametrize("number_of_shells", [1, 2, 3])
def test_get_shifted_relative_coordinates(relative_coordinates, number_of_shells):

    relative_lattice_vectors = _get_relative_coordinates_lattice_vectors(number_of_shells)
    computed_shifted_coordinates = _get_shifted_relative_coordinates(relative_coordinates,
                                                                     relative_lattice_vectors)

    for cell_idx, relative_lattice_vector in enumerate(relative_lattice_vectors):
        expected_cell_shift_coordinates = relative_coordinates + relative_lattice_vector

        computed_cell_shift_coordinates = computed_shifted_coordinates[:, cell_idx, :, :]

        torch.testing.assert_allclose(expected_cell_shift_coordinates, computed_cell_shift_coordinates)


def test_get_shortest_distance_that_crosses_unit_cell(basis_vectors):
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
    computed_shortest_distances = _get_shortest_distance_that_crosses_unit_cell(basis_vectors)

    torch.testing.assert_allclose(expected_shortest_distances, computed_shortest_distances)

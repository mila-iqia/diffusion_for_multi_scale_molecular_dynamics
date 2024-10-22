import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_positions_from_coordinates
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import (
    compute_distances, compute_distances_in_batch,
    get_orthogonal_basis_vectors)


@pytest.fixture()
def spatial_dimension():
    return 3


@pytest.fixture()
def cell_dimensions(spatial_dimension):
    values = []
    for v in list(7.5 + 2.5 * torch.rand(spatial_dimension).numpy()):
        values.append(float(v))
    return values


@pytest.fixture()
def batch_size():
    return 16


@pytest.fixture()
def number_of_atoms():
    return 12


@pytest.fixture()
def relative_coordinates(batch_size, number_of_atoms, spatial_dimension):
    return torch.rand(batch_size, number_of_atoms, spatial_dimension)


def test_get_orthogonal_basis_vectors(batch_size, cell_dimensions):
    computed_basis_vectors = get_orthogonal_basis_vectors(batch_size, cell_dimensions)
    expected_basis_vectors = torch.zeros_like(computed_basis_vectors)

    for d, acell in enumerate(cell_dimensions):
        expected_basis_vectors[:, d, d] = acell
    torch.testing.assert_allclose(computed_basis_vectors, expected_basis_vectors)


def test_compute_distances(batch_size, cell_dimensions, relative_coordinates):
    max_distance = min(cell_dimensions) - 0.5
    basis_vectors = get_orthogonal_basis_vectors(batch_size, cell_dimensions)

    cartesian_positions = get_positions_from_coordinates(
        relative_coordinates=relative_coordinates, basis_vectors=basis_vectors
    )

    distances = compute_distances(
        cartesian_positions=cartesian_positions,
        basis_vectors=basis_vectors,
        max_distance=float(max_distance),
    )

    alt_distances = compute_distances_in_batch(
        cartesian_positions=cartesian_positions,
        unit_cell=basis_vectors,
        max_distance=float(max_distance),
    )

    torch.testing.assert_allclose(distances, alt_distances)

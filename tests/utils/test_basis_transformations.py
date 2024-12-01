import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates, get_reciprocal_basis_vectors,
    get_relative_coordinates_from_cartesian_positions,
    map_axl_composition_to_unit_cell, map_relative_coordinates_to_unit_cell)


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def number_of_atoms():
    return 32


@pytest.fixture
def relative_coordinates(batch_size, number_of_atoms):
    return torch.rand(batch_size, number_of_atoms, 3)


@pytest.fixture
def num_atom_types():
    return 5


def test_get_reciprocal_basis_vectors(basis_vectors):
    reciprocal_basis_vectors = get_reciprocal_basis_vectors(basis_vectors)
    assert reciprocal_basis_vectors.shape == basis_vectors.shape

    identity = torch.eye(3)

    for a_matrix, b_matrix in zip(basis_vectors, reciprocal_basis_vectors):
        torch.testing.assert_allclose(a_matrix @ b_matrix, identity)
        torch.testing.assert_allclose(b_matrix @ a_matrix, identity)


def test_get_positions_from_coordinates(
    batch_size, relative_coordinates, basis_vectors
):

    computed_positions = get_positions_from_coordinates(
        relative_coordinates, basis_vectors
    )

    expected_positions = torch.empty(relative_coordinates.shape, dtype=torch.float32)
    for batch_idx, (a1, a2, a3) in enumerate(basis_vectors):
        for pos_idx, (x1, x2, x3) in enumerate(relative_coordinates[batch_idx]):
            expected_positions[batch_idx, pos_idx, :] = x1 * a1 + x2 * a2 + x3 * a3

    torch.testing.assert_close(expected_positions, computed_positions)


def test_get_relative_coordinates_from_cartesian_positions(
    relative_coordinates, basis_vectors
):
    cartesian_positions = get_positions_from_coordinates(
        relative_coordinates, basis_vectors
    )
    reciprocal_basis_vectors = get_reciprocal_basis_vectors(basis_vectors)

    computed_relative_coordinates = get_relative_coordinates_from_cartesian_positions(
        cartesian_positions, reciprocal_basis_vectors
    )

    torch.testing.assert_close(computed_relative_coordinates, relative_coordinates)


def test_remainder_failure():
    # This test demonstrates how torch.remainder does not do what we want, which is why we need
    # to define the function "map_relative_coordinates_to_unit_cell".
    epsilon = -torch.tensor(1.0e-8)
    relative_coordinates_not_in_unit_cell = torch.remainder(epsilon, 1.0)
    assert relative_coordinates_not_in_unit_cell == 1.0


@pytest.mark.parametrize("shape", [(10,), (10, 20), (3, 4, 5)])
def test_map_relative_coordinates_to_unit_cell_hard(shape):
    relative_coordinates = 1e-8 * (torch.rand(shape) - 0.5)
    computed_relative_coordinates = map_relative_coordinates_to_unit_cell(
        relative_coordinates
    )

    positive_relative_coordinates_mask = relative_coordinates >= 0.0
    assert torch.all(
        relative_coordinates[positive_relative_coordinates_mask]
        == computed_relative_coordinates[positive_relative_coordinates_mask]
    )
    torch.testing.assert_close(
        computed_relative_coordinates[~positive_relative_coordinates_mask],
        torch.zeros_like(
            computed_relative_coordinates[~positive_relative_coordinates_mask]
        ),
    )


@pytest.mark.parametrize("shape", [(100, 8, 16)])
def test_map_relative_coordinates_to_unit_cell_easy(shape):
    # Very unlikely to hit the edge cases.
    relative_coordinates = 10.0 * (torch.rand(shape) - 0.5)
    expected_values = torch.remainder(relative_coordinates, 1.0)
    computed_values = map_relative_coordinates_to_unit_cell(relative_coordinates)
    torch.testing.assert_close(computed_values, expected_values)


@pytest.mark.parametrize("shape", [(10,), (10, 20), (3, 4, 5)])
def test_map_axl_to_unit_cell_hard(shape, num_atom_types):
    atom_types = torch.randint(0, num_atom_types + 1, shape)
    relative_coordinates = 1e-8 * (torch.rand(shape) - 0.5)
    axl_composition = AXL(A=atom_types, X=relative_coordinates, L=torch.rand(shape))

    computed_axl_composition = map_axl_composition_to_unit_cell(
        axl_composition, device=torch.device("cpu")
    )

    positive_relative_coordinates_mask = relative_coordinates >= 0.0
    assert torch.all(
        relative_coordinates[positive_relative_coordinates_mask]
        == computed_axl_composition.X[positive_relative_coordinates_mask]
    )
    torch.testing.assert_close(
        computed_axl_composition.X[~positive_relative_coordinates_mask],
        torch.zeros_like(
            computed_axl_composition.X[~positive_relative_coordinates_mask]
        ),
    )
    assert torch.all(computed_axl_composition.A == axl_composition.A)
    assert torch.all(computed_axl_composition.L == axl_composition.L)

import math

import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.utils import \
    get_distances_from_reference_point


@pytest.fixture(params=[2, 3, 30])
def number_of_atoms(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def spatial_dimension(request):
    return request.param


@pytest.fixture
def basis_vectors(spatial_dimension):
    box_size = np.random.random((spatial_dimension,))
    return np.diag(box_size)


@pytest.fixture
def lattice_parameters(spatial_dimension, basis_vectors):
    lp = np.concatenate(
        (
            np.diag(basis_vectors),
            np.zeros(int(spatial_dimension * (spatial_dimension - 1) / 2)),
        )
    )
    return lp


@pytest.fixture
def atom_relative_positions(number_of_atoms, spatial_dimension):
    return np.random.random((number_of_atoms, spatial_dimension))


@pytest.fixture
def atom_cartesian_positions(atom_relative_positions, basis_vectors):
    return np.matmul(atom_relative_positions, basis_vectors)


@pytest.fixture
def reference_point_relative(spatial_dimension):
    return np.random.random((spatial_dimension,))


@pytest.fixture
def reference_point_cartesian(reference_point_relative, basis_vectors):
    return np.matmul(reference_point_relative, basis_vectors)


def test_get_distances_from_reference_point(
    atom_relative_positions,
    reference_point_relative,
    atom_cartesian_positions,
    reference_point_cartesian,
    lattice_parameters,
    basis_vectors,
    spatial_dimension,
):
    expected_distances = []
    box_dimensions = np.diag(basis_vectors)
    for atom in atom_cartesian_positions:
        distance = 0.0
        for d in range(spatial_dimension):
            distance_ = min(
                [
                    (atom[d] - reference_point_cartesian[d]) ** 2,
                    (atom[d] - reference_point_cartesian[d] + box_dimensions[d]) ** 2,
                    (atom[d] - reference_point_cartesian[d] - box_dimensions[d]) ** 2,
                ]
            )
            distance += distance_
        expected_distances.append(math.sqrt(distance))

    calculated_distances = get_distances_from_reference_point(
        atom_relative_positions, reference_point_relative, lattice_parameters
    )

    for ed, cd in zip(expected_distances, calculated_distances):
        assert np.isclose(ed, cd)

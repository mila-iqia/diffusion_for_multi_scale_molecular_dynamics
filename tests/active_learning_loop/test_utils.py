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
def atom_positions(number_of_atoms, spatial_dimension):
    return np.random.random((number_of_atoms, spatial_dimension))


@pytest.fixture
def lattice_parameters(spatial_dimension):
    return np.ones((spatial_dimension,))


@pytest.fixture
def reference_point(spatial_dimension):
    return np.random.random((spatial_dimension,))


def test_get_distances_from_reference_point(
    atom_positions, reference_point, lattice_parameters, spatial_dimension
):
    expected_distances = []
    for atom in atom_positions:
        distance = 0.0
        for d in range(spatial_dimension):
            distance_ = min(
                [
                    (atom[d] - reference_point[d]) ** 2,
                    (atom[d] - reference_point[d] + lattice_parameters[d]) ** 2,
                    (atom[d] - reference_point[d] - lattice_parameters[d]) ** 2,
                ]
            )
            distance += distance_
        expected_distances.append(math.sqrt(distance))

    calculated_distances = get_distances_from_reference_point(
        atom_positions, reference_point, lattice_parameters
    )

    for ed, cd in zip(expected_distances, calculated_distances):
        assert np.isclose(ed, cd)

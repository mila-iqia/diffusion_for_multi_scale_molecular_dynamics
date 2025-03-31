from pathlib import Path

import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.generators.sampling_constraint import (
    SamplingConstraint, read_sampling_constraint, write_sampling_constraint)


@pytest.fixture()
def elements():
    return ["A", "B", "X", "dummy", "test_element"]


@pytest.fixture()
def number_of_atom_types(elements):
    return len(elements)


@pytest.fixture()
def num_classes(number_of_atom_types):
    return number_of_atom_types + 1


@pytest.fixture()
def number_of_atoms():
    return 8


@pytest.fixture()
def spatial_dimensions():
    return 3


@pytest.fixture()
def constrained_indices(number_of_atoms):
    indices = np.arange(number_of_atoms)
    np.random.shuffle(indices)
    return indices[: number_of_atoms // 2]


@pytest.fixture()
def reference_relative_coordinates(number_of_atoms, spatial_dimensions):
    return np.random.rand(number_of_atoms, spatial_dimensions)


@pytest.fixture()
def reference_atom_types(number_of_atoms, elements):
    element_types = ElementTypes(elements)
    random_ints = np.random.randint(0, len(elements), number_of_atoms)
    return np.array([element_types.get_element(i) for i in random_ints])


@pytest.fixture()
def constrained_relative_coordinates(
    reference_relative_coordinates, constrained_indices
):
    return reference_relative_coordinates[constrained_indices]


@pytest.fixture()
def constrained_atom_types(reference_atom_types, constrained_indices):
    return reference_atom_types[constrained_indices]


@pytest.fixture(params=[True, False])
def write_constrained_indices(request):
    return request.param


@pytest.fixture()
def sampling_constraint(
    constrained_relative_coordinates,
    constrained_atom_types,
    constrained_indices,
    write_constrained_indices,
):
    if write_constrained_indices:
        sampling_constraint = SamplingConstraint(
            constrained_relative_coordinates=constrained_relative_coordinates,
            constrained_atom_types=constrained_atom_types,
            constrained_indices=constrained_indices,
        )
    else:
        sampling_constraint = SamplingConstraint(
            constrained_relative_coordinates=constrained_relative_coordinates,
            constrained_atom_types=constrained_atom_types,
        )

    return sampling_constraint


def test_write_and_read_sampling_constraint(
    tmp_path, sampling_constraint, write_constrained_indices
):

    output_path = Path(tmp_path) / "test_sampling_constraint.pkl"
    write_sampling_constraint(sampling_constraint, output_path)
    new_sampling_constraint = read_sampling_constraint(output_path)

    np.testing.assert_allclose(
        sampling_constraint.constrained_relative_coordinates,
        new_sampling_constraint.constrained_relative_coordinates,
    )

    assert len(sampling_constraint.constrained_atom_types) == len(
        new_sampling_constraint.constrained_atom_types
    )

    for symbol1, symbol2 in zip(
        sampling_constraint.constrained_atom_types,
        new_sampling_constraint.constrained_atom_types,
    ):
        assert symbol1 == symbol2

    if write_constrained_indices:
        np.testing.assert_allclose(
            sampling_constraint.constrained_indices,
            new_sampling_constraint.constrained_indices,
        )
    else:
        assert new_sampling_constraint.constrained_indices is None

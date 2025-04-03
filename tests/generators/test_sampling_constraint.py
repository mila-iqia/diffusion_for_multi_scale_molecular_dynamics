from pathlib import Path

import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.sampling_constraint import (
    SamplingConstraint, read_sampling_constraint, write_sampling_constraint)


@pytest.fixture()
def elements():
    return ["A", "B", "X", "dummy", "test_element"]


@pytest.fixture()
def number_of_atom_types(elements):
    return len(elements)


@pytest.fixture()
def number_of_atoms():
    return 32


@pytest.fixture()
def spatial_dimensions():
    return 3


@pytest.fixture()
def constrained_indices(number_of_atoms):
    return torch.randperm(number_of_atoms)[:number_of_atoms // 2]


@pytest.fixture()
def reference_relative_coordinates(number_of_atoms, spatial_dimensions):
    return torch.rand(number_of_atoms, spatial_dimensions)


@pytest.fixture()
def reference_atom_types(number_of_atoms, elements):
    return torch.randint(0, len(elements), (number_of_atoms,))


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
    elements,
    constrained_relative_coordinates,
    constrained_atom_types,
    constrained_indices,
    write_constrained_indices,
):
    if write_constrained_indices:
        sampling_constraint = SamplingConstraint(
            elements=elements,
            constrained_relative_coordinates=constrained_relative_coordinates,
            constrained_atom_types=constrained_atom_types,
            constrained_indices=constrained_indices,
        )
    else:
        sampling_constraint = SamplingConstraint(
            elements=elements,
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

    assert new_sampling_constraint.elements == sampling_constraint.elements

    torch.testing.assert_close(
        sampling_constraint.constrained_relative_coordinates,
        new_sampling_constraint.constrained_relative_coordinates,
    )

    torch.testing.assert_close(
        sampling_constraint.constrained_atom_types,
        new_sampling_constraint.constrained_atom_types,
    )

    if write_constrained_indices:
        torch.testing.assert_close(
            sampling_constraint.constrained_indices,
            new_sampling_constraint.constrained_indices,
        )
    else:
        assert new_sampling_constraint.constrained_indices is None

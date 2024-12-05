from pathlib import Path

import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.sampling_constraint import (
    SamplingConstraint, read_sampling_constraint, write_sampling_constraint)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@pytest.fixture()
def total_time_steps():
    return 10


@pytest.fixture()
def starting_free_diffusion_time_step():
    return 5


@pytest.fixture()
def number_of_atom_classes():
    return 4


@pytest.fixture()
def number_of_atoms():
    return 8


@pytest.fixture()
def spatial_dimensions():
    return 3


@pytest.fixture()
def constrained_atom_indices(number_of_atoms):
    return torch.randperm(number_of_atoms)[: number_of_atoms // 2].to(torch.long)


@pytest.fixture()
def reference_relative_coordinates(
    total_time_steps, number_of_atoms, spatial_dimensions
):
    return torch.rand(total_time_steps, number_of_atoms, spatial_dimensions)


@pytest.fixture()
def reference_atom_types(total_time_steps, number_of_atoms, number_of_atom_classes):
    return torch.randint(0, number_of_atom_classes, (total_time_steps, number_of_atoms))


@pytest.fixture()
def reference_lattice(total_time_steps, spatial_dimensions):
    return torch.rand(total_time_steps, spatial_dimensions, spatial_dimensions)


@pytest.fixture()
def reference_compositions(
    reference_relative_coordinates, reference_atom_types, reference_lattice
):
    return AXL(
        A=reference_atom_types, X=reference_relative_coordinates, L=reference_lattice
    )


@pytest.fixture()
def sampling_constraint(
    total_time_steps,
    starting_free_diffusion_time_step,
    reference_compositions,
    constrained_atom_indices,
):

    sampling_constraints = SamplingConstraint(
        total_time_steps=total_time_steps,
        starting_free_diffusion_time_step=starting_free_diffusion_time_step,
        reference_compositions=reference_compositions,
        constrained_atom_indices=constrained_atom_indices,
    )
    return sampling_constraints


def test_write_sampling_constraint(tmp_path, sampling_constraint):
    output_path = Path(tmp_path) / "test_sampling_constraint.pkl"
    write_sampling_constraint(sampling_constraint, output_path)
    new_sampling_constraint = read_sampling_constraint(output_path)

    assert (
        new_sampling_constraint.total_time_steps == sampling_constraint.total_time_steps
    )
    assert (
        new_sampling_constraint.starting_free_diffusion_time_step
        == sampling_constraint.starting_free_diffusion_time_step
    )

    torch.testing.assert_close(
        sampling_constraint.constrained_atom_indices,
        new_sampling_constraint.constrained_atom_indices,
    )

    torch.testing.assert_close(
        sampling_constraint.reference_compositions,
        new_sampling_constraint.reference_compositions,
    )

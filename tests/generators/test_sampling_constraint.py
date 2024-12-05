from pathlib import Path

import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.sampling_constraint import (
    SamplingConstraintParameters, create_sampling_constraint,
    read_sampling_constraint, write_sampling_constraint)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters


@pytest.fixture()
def total_time_steps():
    return 10


@pytest.fixture()
def starting_free_diffusion_time_step():
    return 5


@pytest.fixture()
def number_of_atom_types():
    return 4


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
def noise_parameters(total_time_steps):
    return NoiseParameters(total_time_steps=total_time_steps)


@pytest.fixture()
def constrained_atom_indices(number_of_atoms):
    return torch.randperm(number_of_atoms)[: number_of_atoms // 2].to(torch.long)


@pytest.fixture()
def reference_relative_coordinates(number_of_atoms, spatial_dimensions):
    return torch.rand(number_of_atoms, spatial_dimensions)


@pytest.fixture()
def reference_atom_types(number_of_atoms, number_of_atom_types):
    return torch.randint(0, number_of_atom_types, (number_of_atoms, ))


@pytest.fixture()
def reference_lattice(spatial_dimensions):
    return torch.rand(spatial_dimensions, spatial_dimensions)


@pytest.fixture()
def reference_composition(
    reference_relative_coordinates, reference_atom_types, reference_lattice
):
    return AXL(
        A=reference_atom_types, X=reference_relative_coordinates, L=reference_lattice
    )


@pytest.fixture()
def sampling_constraint_parameters(noise_parameters, number_of_atom_types, starting_free_diffusion_time_step,
                                   constrained_atom_indices, reference_composition):

    return SamplingConstraintParameters(noise_parameters=noise_parameters,
                                        num_atom_types=number_of_atom_types,
                                        starting_free_diffusion_time_step=starting_free_diffusion_time_step,
                                        constrained_atom_indices=constrained_atom_indices,
                                        reference_composition=reference_composition)


def test_sampling_constraint_parameters(sampling_constraint_parameters, reference_composition):
    torch.testing.assert_close(sampling_constraint_parameters.reference_composition, reference_composition)


def test_create_sampling_constraint(sampling_constraint_parameters, reference_composition, num_classes):
    sampling_constraint = create_sampling_constraint(sampling_constraint_parameters)

    # Check that the t=0 composition is consistent with the reference composition.
    torch.testing.assert_close(sampling_constraint.constraint_compositions.X[0], reference_composition.X)
    torch.testing.assert_close(sampling_constraint.constraint_compositions.A[0], reference_composition.A)
    torch.testing.assert_close(sampling_constraint.constraint_compositions.L[0], reference_composition.L)

    # Check that the noised atom types are either the reference or MASK
    reference_a = einops.repeat(reference_composition.A, "natoms -> ntimes natoms",
                                ntimes=sampling_constraint_parameters.noise_parameters.total_time_steps)
    noised_a = sampling_constraint.constraint_compositions.A

    masked_class_index = num_classes - 1
    same_class = noised_a == reference_a
    mask_class = noised_a == masked_class_index

    assert torch.logical_or(same_class, mask_class).all()


def test_write_sampling_constraint(tmp_path, sampling_constraint_parameters):

    sampling_constraint = create_sampling_constraint(sampling_constraint_parameters)

    output_path = Path(tmp_path) / "test_sampling_constraint.pkl"
    write_sampling_constraint(sampling_constraint, output_path)
    new_sampling_constraint = read_sampling_constraint(output_path)

    new_sampling_constraint_parameters = new_sampling_constraint.sampling_constraint_parameters

    assert new_sampling_constraint_parameters.num_atom_types == sampling_constraint_parameters.num_atom_types

    assert (new_sampling_constraint_parameters.starting_free_diffusion_time_step
            == sampling_constraint_parameters.starting_free_diffusion_time_step)

    torch.testing.assert_close(
        sampling_constraint.sampling_constraint_parameters.constrained_atom_indices,
        new_sampling_constraint.sampling_constraint_parameters.constrained_atom_indices,
    )

    torch.testing.assert_close(
        sampling_constraint.constraint_compositions,
        new_sampling_constraint.constraint_compositions,
    )

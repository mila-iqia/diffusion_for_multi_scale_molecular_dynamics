from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from ase import Atoms

from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    SinglePointCalculation
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_positions_from_coordinates
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import (
    compute_distances, compute_distances_in_batch, create_perturbed_structures,
    get_orthogonal_basis_vectors, label_configurations)


@pytest.fixture()
def seed_structure():
    return Atoms(
        "Si2Ge2",
        positions=[[0.0, 0.0, 0.0], [1.35, 1.35, 1.35], [2.7, 2.7, 0.0], [0.0, 2.7, 2.7]],
        cell=5.43 * np.eye(3),
        pbc=True,
    )


def test_create_perturbed_structures_count_for_single_structure(seed_structure):
    """A single structure yields exactly number_of_configurations perturbed copies."""
    perturbed_structures = create_perturbed_structures(seed_structure, standard_deviation=0.1,
                                                       number_of_configurations=5)
    assert len(perturbed_structures) == 5


def test_create_perturbed_structures_count_for_list(seed_structure):
    """A list of structures yields number_of_configurations copies per input structure."""
    perturbed_structures = create_perturbed_structures([seed_structure, seed_structure], standard_deviation=0.1,
                                                       number_of_configurations=3)
    assert len(perturbed_structures) == 6


def test_create_perturbed_structures_perturbs_positions_only(seed_structure):
    """Each copy displaces the atoms (by roughly the std) while keeping the cell, species and atom count."""
    standard_deviation = 0.1
    perturbed_structures = create_perturbed_structures(seed_structure, standard_deviation=standard_deviation,
                                                       number_of_configurations=250)

    for structure in perturbed_structures:
        assert len(structure) == len(seed_structure)
        assert structure.get_chemical_symbols() == seed_structure.get_chemical_symbols()
        np.testing.assert_allclose(np.asarray(structure.cell), np.asarray(seed_structure.cell))
        assert not np.allclose(structure.get_positions(), seed_structure.get_positions())

    # The per-coordinate displacements are Gaussian(0, standard_deviation); check the sample statistics.
    displacements = np.array(
        [structure.get_positions() - seed_structure.get_positions() for structure in perturbed_structures]
    )
    np.testing.assert_allclose(displacements.mean(), 0.0, atol=0.02)
    np.testing.assert_allclose(displacements.std(), standard_deviation, rtol=0.15)


def test_create_perturbed_structures_does_not_mutate_input(seed_structure):
    """The seed structure is copied, not rattled in place."""
    original_positions = seed_structure.get_positions().copy()
    create_perturbed_structures(seed_structure, standard_deviation=0.1, number_of_configurations=3)
    np.testing.assert_allclose(seed_structure.get_positions(), original_positions)


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


def test_label_configurations_labels_via_the_single_point_calculator():
    """label_configurations returns the configurations carrying the calculator's energy and forces."""
    configuration = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.35, 1.35, 1.35]], cell=5.43 * np.eye(3), pbc=True)
    single_point_calculator = MagicMock()
    single_point_calculator.calculate_many.side_effect = lambda list_atoms: [
        SinglePointCalculation(calculation_type="stub", atoms=atoms,
                               forces=np.zeros((len(atoms), 3)), energy=-1.5)
        for atoms in list_atoms
    ]

    labelled_configurations = label_configurations([configuration], single_point_calculator)

    assert len(labelled_configurations) == 1
    assert labelled_configurations[0].get_potential_energy() == -1.5
    assert labelled_configurations[0].get_forces().shape == (2, 3)

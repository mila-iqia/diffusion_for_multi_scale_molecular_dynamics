import numpy as np
import pytest
from crystal_diffusion.oracle.lammps import get_energy_and_forces_from_lammps


@pytest.fixture
def high_symmetry_lattice():
    box = np.eye(3) * 4
    return box


@pytest.fixture
def high_symmetry_positions():
    positions = np.array([[0, 0, 0], [2, 2, 2]])
    return positions


# do not run on github because no lammps
@pytest.mark.not_on_github
def test_high_symmetry(high_symmetry_positions, high_symmetry_lattice):
    energy, forces = get_energy_and_forces_from_lammps(high_symmetry_positions, high_symmetry_lattice,
                                                       atom_types=np.array([1, 1]))
    for x in ['x', 'y', 'z']:
        assert np.allclose(forces[f'f{x}'], [0, 0])
    assert energy < 0


@pytest.fixture
def low_symmetry_positions():
    positions = np.array([[0.23, 1.2, 2.01], [3.2, 0.9, 3.87]])
    return positions


@pytest.mark.not_on_github
def test_low_symmetry(low_symmetry_positions, high_symmetry_lattice):
    energy, forces = get_energy_and_forces_from_lammps(low_symmetry_positions, high_symmetry_lattice,
                                                       atom_types=np.array([1, 1]))
    for x in ['x', 'y', 'z']:
        assert not np.allclose(forces[f'f{x}'], [0, 0])
    assert energy < 0

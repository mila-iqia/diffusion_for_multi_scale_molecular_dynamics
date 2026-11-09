"""Tests for the general io helpers (ASE .traj round-trip)."""

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from diffusion_for_multi_scale_molecular_dynamics.io.utils import (
    read_atoms_trajectory, write_atoms_trajectory)


def _labelled_atoms() -> Atoms:
    atoms = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.1, 1.1, 1.1]], cell=[5.0, 5.0, 5.0], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=-3.14, forces=np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]))
    atoms.info["active_environment_indices"] = np.array([0], dtype=int)
    return atoms


def test_trajectory_round_trip_preserves_energy_forces_and_info(tmp_path):
    """Writing then reading a .traj keeps the calculator's energy/forces and the info dict entries."""
    original = _labelled_atoms()
    path = tmp_path / "frames.traj"

    write_atoms_trajectory([original], path)
    reloaded = read_atoms_trajectory(path)

    assert len(reloaded) == 1
    assert reloaded[0].get_potential_energy() == -3.14
    assert np.allclose(reloaded[0].get_forces(), original.get_forces())
    assert np.array_equal(reloaded[0].info["active_environment_indices"], np.array([0]))


def test_trajectory_round_trip_preserves_info_arrays(tmp_path):
    """Per-atom data stored in the info dict (e.g. an uncertainty vector) survives the round-trip."""
    atoms = Atoms("Si3", positions=np.eye(3), cell=[6.0, 6.0, 6.0], pbc=True)
    atoms.info["uncertainty"] = np.array([0.7, 0.9, 0.5])
    path = tmp_path / "uncertain.traj"

    write_atoms_trajectory([atoms], path)
    reloaded = read_atoms_trajectory(path)

    assert np.allclose(reloaded[0].info["uncertainty"], [0.7, 0.9, 0.5])


def test_trajectory_round_trip_preserves_multiple_frames(tmp_path):
    """All frames are written and read back in order."""
    frames = [_labelled_atoms() for _ in range(3)]
    path = tmp_path / "many.traj"

    write_atoms_trajectory(frames, path)
    reloaded = read_atoms_trajectory(path)

    assert len(reloaded) == 3

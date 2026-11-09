"""Read a real LAMMPS text dump (a committed reference file) and check every field is recovered.

The reference dump was produced by LAMMPS for a genuinely triclinic (60-degree) 16-atom silicon cell, so
it guards the ase-based reader against the box being misread as orthogonal.
"""

from pathlib import Path

import numpy as np
from ase.build import bulk

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.outputs import \
    extract_all_fields_from_dump

REFERENCE_DIRECTORY = Path(__file__).parents[2] / "reference_files" / "lammps"
TRICLINIC_DUMP = REFERENCE_DIRECTORY / "triclinic_single_point.dump"
ORTHOGONAL_DUMP = REFERENCE_DIRECTORY / "orthogonal_single_point.dump"


def _expected_triclinic_atoms():
    """The exact configuration the triclinic reference dump was generated from."""
    return bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))


def test_reader_recovers_the_triclinic_structure():
    """The dump reads back to the known triclinic structure: cell, positions and species all match."""
    expected_atoms = _expected_triclinic_atoms()

    list_atoms, _, _ = extract_all_fields_from_dump(TRICLINIC_DUMP)

    assert len(list_atoms) == 1
    atoms = list_atoms[0]

    # The box is triclinic (all angles 60 degrees) and must be read as such, not collapsed to orthogonal.
    np.testing.assert_allclose(atoms.cell.angles(), 60.0, atol=1e-4)
    np.testing.assert_allclose(atoms.cell.cellpar(), expected_atoms.cell.cellpar(), atol=1e-4)

    # Same atoms: species and (minimum-image) fractional coordinates.
    assert atoms.get_chemical_symbols() == ["Si"] * len(expected_atoms)
    fractional_difference = (
        atoms.get_scaled_positions() - expected_atoms.get_scaled_positions() + 0.5
    ) % 1.0 - 0.5
    np.testing.assert_allclose(fractional_difference, 0.0, atol=1e-4)


def test_reader_recovers_forces_and_uncertainties():
    """Forces and the per-atom uncertainty column are read with the right shape and finite values."""
    number_of_atoms = len(_expected_triclinic_atoms())
    _, list_forces, list_uncertainties = extract_all_fields_from_dump(TRICLINIC_DUMP)

    assert list_forces[0].shape == (number_of_atoms, 3)
    assert np.all(np.isfinite(list_forces[0]))

    assert list_uncertainties[0].shape == (number_of_atoms,)
    assert np.all(np.isfinite(list_uncertainties[0]))


def test_reader_reads_the_expected_per_atom_uncertainties():
    """The per-atom uncertainty column is read value-for-value (the fixture set c_unc_at to the atom id)."""
    _, _, list_uncertainties = extract_all_fields_from_dump(ORTHOGONAL_DUMP)

    number_of_atoms = 8  # the orthogonal reference cell has 8 atoms
    expected_uncertainties = np.arange(1, number_of_atoms + 1)  # c_unc_at was set to the atom id
    np.testing.assert_allclose(list_uncertainties[0], expected_uncertainties)


def test_reader_returns_none_uncertainty_when_field_absent():
    """Requesting a per-atom column the dump does not contain yields None (not an error)."""
    _, _, list_uncertainties = extract_all_fields_from_dump(TRICLINIC_DUMP, uncertainty_field="c_not_a_field")
    assert list_uncertainties[0] is None

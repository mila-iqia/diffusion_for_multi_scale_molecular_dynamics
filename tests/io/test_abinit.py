"""Tests for the Abinit io layer (input writing, output + pseudopotential parsing).

The reference files under reference_files/abinit are genuine Abinit files (an MgH2 run from an earlier
project), used here only to exercise the parsers on real content.
"""

import pytest
from ase import Atoms

from diffusion_for_multi_scale_molecular_dynamics.io.abinit import (
    read_abinit_output, read_exchange_correlation_from_pseudopotential,
    write_abinit_input)


@pytest.fixture
def abinit_reference_directory(reference_files_directory):
    return reference_files_directory / "abinit"


class TestReadExchangeCorrelation:
    def test_reads_pspxc_from_pseudopotentials(self, abinit_reference_directory):
        """pspxc is read from the header of each real ONCVPSP pseudopotential (both PBE, code 11)."""
        assert read_exchange_correlation_from_pseudopotential(abinit_reference_directory / "Mg.psp8") == 11
        assert read_exchange_correlation_from_pseudopotential(abinit_reference_directory / "H.psp8") == 11

    def test_raises_when_pspxc_absent(self, tmp_path):
        """A file without a 'pspxc' header line is rejected with a clear error."""
        pseudopotential = tmp_path / "not_a_pseudo.psp8"
        pseudopotential.write_text("some header\nwithout the expected line\n")
        with pytest.raises(ValueError, match="pspxc"):
            read_exchange_correlation_from_pseudopotential(pseudopotential)


class TestReadOutput:
    def test_reads_energy_and_forces(self, abinit_reference_directory):
        """The real MgH2 'abinit.abo' parses to its total energy and per-atom forces (no stress here)."""
        energy, forces, stress = read_abinit_output(abinit_reference_directory)

        assert energy == pytest.approx(-26375.690423376)
        assert forces.shape == (48, 3)
        assert stress is None


class TestWriteInput:
    @pytest.fixture
    def atoms(self):
        return Atoms("MgH2", positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5], [3.0, 3.0, 3.0]],
                     cell=[6.0, 6.0, 6.0], pbc=True)

    @pytest.fixture
    def pseudopotentials(self, abinit_reference_directory):
        return {"Mg": abinit_reference_directory / "Mg.psp8", "H": abinit_reference_directory / "H.psp8"}

    def test_writes_input_and_copies_pseudopotentials(self, atoms, pseudopotentials, tmp_path):
        """The input is written (energies tagged eV per ASE) and each pseudopotential is copied next to it."""
        input_path = write_abinit_input(atoms, {"ecut": 300, "ixc": 11}, pseudopotentials, tmp_path)

        assert input_path == tmp_path / "abinit.abi"
        assert "ecut 300 eV" in input_path.read_text()  # ASE tags energies in eV, not Hartree.
        assert (tmp_path / "Mg.psp8").is_file()
        assert (tmp_path / "H.psp8").is_file()

    def test_defaults_ixc_from_pseudopotential(self, atoms, pseudopotentials, tmp_path):
        """With no 'ixc' given, it is set from the pseudopotentials' pspxc (11)."""
        input_path = write_abinit_input(atoms, {"ecut": 300}, pseudopotentials, tmp_path)
        assert "ixc 11" in input_path.read_text()

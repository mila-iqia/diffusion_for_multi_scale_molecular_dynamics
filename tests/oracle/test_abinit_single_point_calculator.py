"""Tests for the AbinitSinglePointCalculator.

The orchestration test stubs the runner (dropping the genuine MgH2 reference output) so the full
write -> run -> read -> clean path is exercised without Abinit. The end-to-end test runs the real binary
and is skipped unless 'abinit' is on PATH.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from diffusion_for_multi_scale_molecular_dynamics.oracle.abinit_runner import \
    AbinitRunner
from diffusion_for_multi_scale_molecular_dynamics.oracle.abinit_single_point_calculator import \
    AbinitSinglePointCalculator


@pytest.fixture
def abinit_reference_directory(reference_files_directory):
    return reference_files_directory / "abinit"


@pytest.fixture
def pseudopotentials(abinit_reference_directory):
    return {"Mg": abinit_reference_directory / "Mg.psp8", "H": abinit_reference_directory / "H.psp8"}


class _StubRunner:
    """A runner that fakes an Abinit run by dropping the reference output and a dummy (huge) WFK file."""

    def __init__(self, reference_output_path: Path):
        self._reference_output_path = reference_output_path

    def run(self, working_directory, input_file_name="abinit.abi"):
        shutil.copy(self._reference_output_path, Path(working_directory) / "abinit.abo")
        (Path(working_directory) / "abinito_WFK").write_text("a very large wavefunction")


class TestOrchestration:
    def test_calculate_writes_input_runs_reads_and_cleans(self, abinit_reference_directory,
                                                          pseudopotentials, tmp_path):
        """calculate() writes the input, runs, parses the energy/forces, keeps outputs, and drops the WFK."""
        atoms = Atoms("MgHH", scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
                      cell=6.0 * np.eye(3), pbc=True)
        runner = _StubRunner(abinit_reference_directory / "abinit.abo")
        calculator = AbinitSinglePointCalculator({"ecut": 300}, pseudopotentials, runner)

        result = calculator.calculate(atoms, results_path=tmp_path / "dump_0.yaml")

        assert result.calculation_type == "abinit"
        assert result.energy == pytest.approx(-26375.690423376)
        assert result.forces.shape == (48, 3)

        # results_path selects a per-sample directory whose files are kept for inspection (minus the WFK).
        run_directory = tmp_path / "dump_0"
        assert (run_directory / "abinit.abi").is_file()
        assert (run_directory / "abinit.abo").is_file()
        assert not (run_directory / "abinito_WFK").exists()


@pytest.mark.requires_abinit
@pytest.mark.slow
class TestEndToEnd:
    def test_runs_real_abinit(self, pseudopotentials, tmp_path):
        """A real (quick, unconverged) Abinit run returns a finite energy and per-atom forces.

        The parameters are a minimal smoke configuration; tune them for your Abinit build if needed.
        """
        atoms = Atoms("MgHH", scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
                      cell=4.0 * np.eye(3), pbc=True)
        parameters = {"ecut": 200, "nstep": 5, "toldfe": 1e-4,
                      "ngkpt": [1, 1, 1], "nshiftk": 1, "shiftk": [0.0, 0.0, 0.0]}
        calculator = AbinitSinglePointCalculator(parameters, pseudopotentials, AbinitRunner())

        result = calculator.calculate(atoms, results_path=tmp_path / "dump_0.yaml")

        assert result.calculation_type == "abinit"
        assert np.isfinite(result.energy)
        assert result.forces.shape == (3, 3)

"""Tests for the AbinitRunner (command building + subprocess plumbing, no Abinit binary needed).

The run() tests use the 'true'/'false' shell utilities as stand-ins for the Abinit binary, so the
subprocess launch and error handling are exercised without an actual Abinit installation.
"""

import pytest

from diffusion_for_multi_scale_molecular_dynamics.oracle.abinit_runner import (
    ABINIT_ERROR_FILE_NAME, ABINIT_LOG_FILE_NAME, AbinitRunner)


class TestBuildCommand:
    def test_serial_command(self):
        """Without an MPI launcher, Abinit is called directly on the input file."""
        assert AbinitRunner()._build_command("abinit.abi") == ["abinit", "abinit.abi"]

    def test_mpi_command_without_task_count(self):
        """With an MPI launcher but no task count, no '-n' is added."""
        assert AbinitRunner(mpi_runner="mpirun")._build_command("abinit.abi") == ["mpirun", "abinit", "abinit.abi"]

    def test_mpi_command_with_task_count(self):
        """A task count (CPU processes or GPUs) is passed as '-n'."""
        command = AbinitRunner(mpi_runner="mpirun", number_of_mpi_tasks=4)._build_command("abinit.abi")
        assert command == ["mpirun", "-n", "4", "abinit", "abinit.abi"]


class TestRun:
    def test_run_succeeds_and_writes_log_and_error(self, tmp_path):
        """A zero-exit run creates the log and error files in the working directory and does not raise."""
        AbinitRunner(abinit_command="true").run(tmp_path)
        assert (tmp_path / ABINIT_LOG_FILE_NAME).is_file()
        assert (tmp_path / ABINIT_ERROR_FILE_NAME).is_file()

    def test_run_raises_on_failure(self, tmp_path):
        """A non-zero exit is surfaced as a RuntimeError pointing at the log/error files."""
        with pytest.raises(RuntimeError, match="Abinit failed"):
            AbinitRunner(abinit_command="false").run(tmp_path)

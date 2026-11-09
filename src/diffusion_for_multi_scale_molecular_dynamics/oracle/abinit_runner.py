"""Run a single Abinit calculation as a subprocess."""

import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

from diffusion_for_multi_scale_molecular_dynamics.io.abinit import \
    ABINIT_INPUT_FILE_NAME

ABINIT_LOG_FILE_NAME = "abinit.log"
ABINIT_ERROR_FILE_NAME = "abinit.err"


class AbinitRunner:
    """Launch one Abinit calculation, handling the serial / MPI invocation.

    The runner owns *how* Abinit is launched, separate from *what* it computes. When an MPI launcher is
    given, '-n <number_of_mpi_tasks>' is added if a task count is set. For a GPU run, set
    number_of_mpi_tasks to the number of GPUs and set the 'gpu_option' Abinit input variable in the
    calculation parameters.
    """

    def __init__(
        self,
        abinit_command: str = "abinit",
        mpi_runner: Optional[str] = None,
        number_of_mpi_tasks: Optional[int] = None,
    ):
        """Init method.

        Args:
            abinit_command: the command that runs the Abinit binary.
            mpi_runner: the MPI launcher (e.g. 'mpirun'); when None, Abinit is run serially.
            number_of_mpi_tasks: the number of MPI tasks passed as '-n' (i.e. CPU processes or GPUs); when
                None, the MPI launcher is used without '-n'.
        """
        self._abinit_command = abinit_command
        self._mpi_runner = mpi_runner
        self._number_of_mpi_tasks = number_of_mpi_tasks

    def _build_command(self, input_file_name: str) -> List[str]:
        """Build the argument list that runs Abinit on the given input file."""
        abinit_call = f"{self._abinit_command} {input_file_name}"
        if not self._mpi_runner:
            return shlex.split(abinit_call)
        mpi_call = self._mpi_runner
        if self._number_of_mpi_tasks is not None:
            mpi_call = f"{self._mpi_runner} -n {self._number_of_mpi_tasks}"
        return shlex.split(f"{mpi_call} {abinit_call}")

    def run(self, working_directory: Path, input_file_name: str = ABINIT_INPUT_FILE_NAME) -> None:
        """Run Abinit in working_directory; Abinit writes 'abinit.abo' there, stdout/stderr go to log/err.

        Args:
            working_directory: directory holding the input and pseudopotentials; Abinit runs here.
            input_file_name: the Abinit input file to run (defaults to the one write_abinit_input produces).

        Raises:
            RuntimeError: if Abinit exits with a non-zero return code.
        """
        working_directory = Path(working_directory)
        command = self._build_command(input_file_name)
        log_path = working_directory / ABINIT_LOG_FILE_NAME
        error_path = working_directory / ABINIT_ERROR_FILE_NAME

        with open(log_path, "w") as log_file, open(error_path, "w") as error_file:
            completed_process = subprocess.run(
                command, cwd=working_directory, stdout=log_file, stderr=error_file
            )
        if completed_process.returncode != 0:
            raise RuntimeError(
                f"Abinit failed (exit code {completed_process.returncode}). Command: {' '.join(command)}. "
                f"Review {log_path} and {error_path}."
            )

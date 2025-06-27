import os
import subprocess
from pathlib import Path
from typing import Dict, List

_DEFAULT_LAMMPS_CONFIG = dict(mpi_processors=1, openmp_threads=1)


def instantiate_lammps_runner(lammps_executable_path: Path, configuration_dict: Dict):
    """Instantiate lammps runner.

    Args:
        lammps_executable_path: Path to lammps executable.
        configuration_dict: Global configuration dictionary, which can optionally contain LAMMPS instantiation
            parameters.

    Returns:
        lammps_runner: a Lammps runner.
    """
    lammps_config = configuration_dict.get("lammps", _DEFAULT_LAMMPS_CONFIG)
    lammps_runner = LammpsRunner(
        lammps_executable_path=lammps_executable_path,
        mpi_processors=lammps_config["mpi_processors"],
        openmp_threads=lammps_config["openmp_threads"],
    )
    return lammps_runner


class LammpsRunner:
    """LAMMPS Runner.

    This class is responsible for invoking LAMMPS.

    There is a class in pymatgen, pymatgen.io.lammps.utils.LammpsRunner, to drive LAMMPS.
    Here we prefer to keep more control over the execution. In particular, we may want to use
    mpirun, or to not have lammps in the path.
    """

    def __init__(self, lammps_executable_path: Path, mpi_processors: int = 1, openmp_threads: int = 1):
        """Init method.

        Args:
            lammps_executable_path: path to the LAMMPS executable.
            mpi_processors: number of processors to use. Defaults to 1, mpirun is not used.
            openmp_threads: number of OpenMP threads to use per processor. Defaults to 1.
        """
        assert (
            lammps_executable_path.is_file()
        ), f"The path {lammps_executable_path} does not exist."
        self._lammps_executable_path = lammps_executable_path

        self._mpi_processors = mpi_processors
        self._openmp_threads = openmp_threads

    def _build_commands(self, input_file_name: str) -> List[str]:
        """Build the actual command to run."""
        commands = ["mpirun", "-np", f"{self._mpi_processors}", str(self._lammps_executable_path), 
                    "-echo", "none", "-screen", "none", "-i", input_file_name]
        return commands

    def run_lammps(self, working_directory: Path, lammps_input_file_name: str):
        """Run lammps.

        Args:
            working_directory: directory where the LAMMPS job will be executed. It is assumed that all needed files
                are present.
            lammps_input_file_name: name of the lammps input script.
        """
        commands = self._build_commands(lammps_input_file_name)
        environment_variables = os.environ.copy()
        environment_variables['OMP_NUM_THREADS'] = f"{self._openmp_threads}"

        subprocess.run(commands,
                       cwd=working_directory,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       text=True,
                       check=True,
                       env=environment_variables)

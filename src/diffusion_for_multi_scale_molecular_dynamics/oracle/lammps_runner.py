import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

_DEFAULT_LAMMPS_CONFIG = dict(
    mpi_processors=1, openmp_threads=1, mpi_executable="mpirun"
)


def _styles_in_help_section(help_text: str, section: str) -> List[str]:
    """Return the style names listed under a '* <section>:' header in a LAMMPS '-h' output.

    Sections start with a line like '* Pair styles:' and run until the next '* ' header.
    """
    styles: List[str] = []
    inside_section = False
    for line in help_text.splitlines():
        if line.startswith("* "):
            inside_section = line[2:].rstrip(":").strip() == section
            continue
        if inside_section:
            styles.extend(line.split())
    return styles


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
    lammps_runner = SubprocessLammpsRunner(
        lammps_executable_path=lammps_executable_path,
        mpi_processors=lammps_config["mpi_processors"],
        openmp_threads=lammps_config["openmp_threads"],
        mpi_executable=lammps_config.get("mpi_executable", "mpirun"),
    )
    return lammps_runner


class SubprocessLammpsRunner:
    """LAMMPS Runner from an external executable.

    Invoke LAMMPS through an external executable (subprocess), supporting mpirun and OpenMP.
    """

    def __init__(
        self,
        lammps_executable_path: Path,
        mpi_processors: int = 1,
        openmp_threads: int = 1,
        mpi_executable: str = "mpirun",
    ):
        """Init method.

        Args:
            lammps_executable_path: path to the LAMMPS executable.
            mpi_processors: number of processors to use. When 1, LAMMPS runs directly without the MPI launcher.
            openmp_threads: number of OpenMP threads to use per processor. Defaults to 1.
            mpi_executable: the MPI launcher to use when running on more than one processor. Defaults to mpirun.
        """
        assert (
            lammps_executable_path.is_file()
        ), f"The path {lammps_executable_path} does not exist."
        self._lammps_executable_path = lammps_executable_path

        self._mpi_processors = mpi_processors
        self._openmp_threads = openmp_threads
        self._mpi_executable = mpi_executable

    def _build_commands(self, input_file_name: str) -> List[str]:
        """Build the actual command to run."""
        # We do not pass '-screen none': the screen output carries LAMMPS' own error messages, and we capture
        # it so it can be surfaced when the run fails.
        lammps_call = [
            str(self._lammps_executable_path),
            "-echo", "none", "-i", input_file_name,
        ]
        if self._mpi_processors == 1:
            return lammps_call
        return [self._mpi_executable, "-n", f"{self._mpi_processors}"] + lammps_call

    def run_lammps(self, working_directory: Path, lammps_input_file_name: str):
        """Run lammps.

        Args:
            working_directory: directory where the LAMMPS job will be executed. It is assumed that all needed files
                are present.
            lammps_input_file_name: name of the lammps input script.
        """
        commands = self._build_commands(lammps_input_file_name)
        environment_variables = os.environ.copy()
        environment_variables["OMP_NUM_THREADS"] = f"{self._openmp_threads}"

        result = subprocess.run(
            commands,
            cwd=working_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment_variables,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"LAMMPS failed (exit code {result.returncode}). Command: {' '.join(commands)}\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )

    def check_dependency(self, section: str, to_find: str) -> None:
        """Raise if `to_find` is not listed under `section` (e.g. 'Pair styles') in the executable's help output."""
        help_text = subprocess.run(
            [str(self._lammps_executable_path), "-h"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout
        if to_find not in _styles_in_help_section(help_text, section):
            raise RuntimeError(
                f"'{to_find}' is not available in the '{section}' of the LAMMPS executable "
                f"'{self._lammps_executable_path}'."
            )


class InProcessLammpsRunner:
    """In-process LAMMPS Runner.

    Run lammps from the lammps python package, reducing process overhead compared to SubprocessLammpsRunner.
    """

    def run_lammps(
        self, working_directory: Union[Path, str], lammps_input_file_name: str
    ):
        """Run lammps in-process, from within the working directory so relative paths resolve there.

        Args:
            working_directory: directory where the LAMMPS job runs; all needed files must be present.
            lammps_input_file_name: name of the lammps input script.
        """
        import lammps  # Lazy import: the Python binding is an optional dependency.

        original_directory = Path.cwd()
        try:
            os.chdir(working_directory)
            lmp = lammps.lammps(
                cmdargs=["-log", "none", "-echo", "none", "-screen", "none"]
            )
            try:
                lmp.file(lammps_input_file_name)
            finally:
                lmp.close()
        finally:
            os.chdir(original_directory)

    def check_dependency(self, section: str, to_find: str) -> None:
        """Raise if `to_find` is not among the binding's styles for `section` (e.g. 'Pair styles' -> 'pair')."""
        import lammps  # Lazy import: the Python binding is an optional dependency.

        style_category = section.split()[0].lower()
        lmp = lammps.lammps(cmdargs=["-log", "none", "-echo", "none", "-screen", "none"])
        try:
            available_styles = lmp.available_styles(style_category)
        finally:
            lmp.close()
        if to_find not in available_styles:
            raise RuntimeError(
                f"'{to_find}' is not available in the '{section}' of the in-process LAMMPS binding."
            )


def create_lammps_runner(
    lammps_bin: Optional[Path] = None,
    mpi_processors: int = 1,
    openmp_threads: int = 1,
    mpi_executable: str = "mpirun",
) -> Union[SubprocessLammpsRunner, InProcessLammpsRunner]:
    """Create a LAMMPS runner: in-process if lammps_bin is None, else a subprocess against lammps_bin.

    Args:
        lammps_bin: path to a LAMMPS executable, or None to use the in-process Python binding.
        mpi_processors: number of MPI processors (subprocess runner only).
        openmp_threads: number of OpenMP threads per processor (subprocess runner only).
        mpi_executable: the MPI launcher to use (subprocess runner only). Defaults to mpirun.

    Returns:
        lammps_runner: a runner exposing run_lammps(working_directory, lammps_input_file_name).
    """
    if lammps_bin is None:
        return InProcessLammpsRunner()
    return SubprocessLammpsRunner(
        lammps_executable_path=lammps_bin,
        mpi_processors=mpi_processors,
        openmp_threads=openmp_threads,
        mpi_executable=mpi_executable,
    )

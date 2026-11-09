"""Base class for the dynamic drivers (ARTn, MD) that search for uncertain structures with LAMMPS."""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ase import Atoms
from ase.io.lammpsdata import write_lammps_data

from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.dynamic_driver_lammps_input import \
    build_dynamic_driver_lammps_inputs
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.inputs import \
    generate_named_elements_blocks
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    DUMP_FILENAME, INITIAL_CONFIGURATION_FILENAME, LAMMPS_INPUT_FILENAME)
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    configure_logging
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import \
    assert_orthogonal_cell


class DynamicDriver(ABC):
    """Drive a LAMMPS run that searches for an uncertain structure and halts when one is found.

    The shared LAMMPS script (read configuration, define the potential, watch the per-atom uncertainty and
    dump the structure when it exceeds the threshold) lives in the base; subclasses provide the dynamics
    block (ARTn minimization vs MD) and decide the resulting CalculationState.
    """

    def __init__(
        self,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
        initial_configuration: Atoms,
    ):
        """Init method.

        Args:
            lammps_runner: a runner that drives LAMMPS (must handle the MLIP pair_style + uncertainty).
            initial_configuration: the single starting configuration the dynamics is launched from.
        """
        assert_orthogonal_cell(initial_configuration)
        self.initial_configuration = initial_configuration
        self._current_configuration = initial_configuration  # the initial configuration for the next launch
        self._lammps_runner = lammps_runner
        self._lammps_input_filename = LAMMPS_INPUT_FILENAME

    @property
    def maximum_number_of_steps(self) -> Optional[int]:
        """The run's step budget, or None when the driver has no fixed maximum (e.g. ARTn)."""
        return None

    def summarize_interruption(self, working_directory: Path) -> List[str]:
        """Driver-specific log lines for an interrupted run (empty by default; ARTn overrides)."""
        return []

    def summarize_success(self, working_directory: Path) -> List[str]:
        """Driver-specific log lines for a successful run (empty by default; ARTn overrides)."""
        return []

    def run(self, mlip: BaseMLIP, working_directory: Path, uncertainty_threshold: float) -> CalculationState:
        """Run the dynamics with the MLIP and return the resulting calculation state.

        Args:
            mlip: the machine-learning interatomic potential; its deployed LAMMPS potential provides the
                interaction commands and the per-atom uncertainty field.
            working_directory: directory where the run is performed (must not already exist).
            uncertainty_threshold: the run halts when the per-atom uncertainty exceeds this value.

        Returns:
            calculation_state: SUCCESS, INTERRUPTION or ERROR.
        """
        logger = self._setup_working_directory(working_directory)
        parameters = self._build_lammps_parameters(mlip, uncertainty_threshold)
        self._write_lammps_input(working_directory, parameters)
        return self._run_dynamics(working_directory, logger)

    def _run_dynamics(self, working_directory: Path, logger: logging.Logger) -> CalculationState:
        """Run the dynamics once and classify the result (ARTn overrides this to run several searches)."""
        logger.info("Launching LAMMPS")
        execution_time, succeeded = self._execute_lammps(working_directory, logger)
        if not succeeded:
            return CalculationState.ERROR
        logger.info(f"LAMMPS execution has finished. Execution Time: {execution_time: 6.3e} sec.")
        return self._get_calculation_state(working_directory)

    def _setup_working_directory(self, working_directory: Path) -> logging.Logger:
        """Create the (empty) working directory, set up logging, and write the starting configuration."""
        if working_directory.is_dir() and any(working_directory.iterdir()):
            raise ValueError(
                f"The working directory {working_directory} already exists and is not empty! "
                "Exiting to avoid writing over existing data."
            )
        working_directory.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("dynamic_driver_run")
        configure_logging(experiment_dir=str(working_directory), logger=logger, log_to_console=False)

        self._write_configuration(self._current_configuration, working_directory)
        self._prepare_reference_files(working_directory)
        return logger

    def _write_configuration(self, configuration: Atoms, working_directory: Path) -> None:
        """Write a configuration to the LAMMPS data file the dynamics reads (initial_configuration.dat)."""
        _, _, elements_string = generate_named_elements_blocks(configuration)
        with open(working_directory / INITIAL_CONFIGURATION_FILENAME, "w") as file_descriptor:
            # specorder mirrors the (mass-sorted) group/mass blocks so the atom-type integers stay consistent.
            write_lammps_data(file_descriptor, configuration, atom_style="atomic",
                              specorder=elements_string.split(), masses=True)

    def _build_lammps_parameters(self, mlip: BaseMLIP, uncertainty_threshold: float) -> dict:
        """Assemble the template substitution parameters from the MLIP potential and the dynamics block."""
        potential = mlip.lammps_potential
        group_block, mass_block, elements_string = generate_named_elements_blocks(self.initial_configuration)
        dump_fields = " ".join(potential.dump_fields(with_uncertainty=True))
        return dict(
            configuration_file_path=INITIAL_CONFIGURATION_FILENAME,
            interaction_commands="\n".join(potential.interaction_commands(elements_string, with_uncertainty=True)),
            uncertainty_field=potential.uncertainty_field(),
            dump_fields=dump_fields,
            uncertainty_threshold=f"{uncertainty_threshold:.12f}",
            group_block=group_block,
            mass_block=mass_block,
            elements_string=elements_string,
            dynamics_block=self._dynamics_block(),
            trajectory_dump_block=self._trajectory_dump_block(elements_string, dump_fields),
        )

    def _trajectory_dump_block(self, elements_string: str, dump_fields: str) -> str:
        """LAMMPS commands dumping the full trajectory (overridden to '' by drivers that do not want it)."""
        return "\n".join([
            f"dump dump_id all custom 1 {DUMP_FILENAME} {dump_fields}",
            f"dump_modify dump_id element {elements_string}",
        ])

    def _write_lammps_input(self, working_directory: Path, parameters: dict) -> None:
        """Build the shared LAMMPS input script from the parameters and write it to the working directory."""
        script_content = build_dynamic_driver_lammps_inputs(**parameters)
        with open(working_directory / self._lammps_input_filename, "w") as file_descriptor:
            file_descriptor.write(script_content)

    def _execute_lammps(self, working_directory: Path, logger: logging.Logger) -> Tuple[float, bool]:
        """Run LAMMPS in the working directory, returning (elapsed_seconds, succeeded); logs a failure."""
        start_time = time.time()
        try:
            self._lammps_runner.run_lammps(working_directory=working_directory,
                                           lammps_input_file_name=self._lammps_input_filename)
        except RuntimeError:
            logger.exception("LAMMPS execution failed.")
            return time.time() - start_time, False
        return time.time() - start_time, True

    @abstractmethod
    def _prepare_reference_files(self, working_directory: Path) -> None:
        """Copy any driver-specific reference files into the working directory before the run."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def _dynamics_block(self) -> str:
        """Return the LAMMPS dynamics commands rendered into the shared template's $dynamics_block."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def _get_calculation_state(self, working_directory: Path) -> CalculationState:
        """Determine the CalculationState from the finished run's output."""
        raise NotImplementedError("must be implemented in a child class.")

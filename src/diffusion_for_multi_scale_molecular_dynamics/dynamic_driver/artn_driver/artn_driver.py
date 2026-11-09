"""ARTn dynamic driver."""

import logging
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Union

from ase import Atoms

from diffusion_for_multi_scale_molecular_dynamics.dynamic_driver.base_dynamic_driver import \
    DynamicDriver
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.artn import (
    EIGENVALUE_LOST_MESSAGE, MACRO_STAGE_DESCRIPTIONS,
    MICRO_STAGE_DESCRIPTIONS, append_artn_search_summary,
    build_artn_lammps_tail, collect_artn_run_information,
    collect_artn_transition_information,
    get_calculation_state_from_artn_output, parse_artn_failure_message,
    read_artn_search_summaries, read_artn_xyz, write_artn_input_file)
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.artn_input_configuration import \
    ArtnInputConfiguration
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.inputs import \
    generate_named_elements_blocks
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import \
    configurations_are_equivalent

ARTN_PLUGIN_PATH_ENVIRONMENT_VARIABLE = "ARTN_PLUGIN_PATH"
ARTN_LIBRARY_FILE_NAME = "libartn-lmp.so"


class ArtnDriver(DynamicDriver):
    """Drive an ARTn saddle search with LAMMPS, halting when an uncertain structure is found."""

    def __init__(
        self,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
        initial_configuration: Atoms,
        artn_input_configuration: Optional[ArtnInputConfiguration] = None,
        artn_library_plugin_path: Optional[Path] = None,
        number_of_requested_saddles: int = 1,
        restart_from_new_min: bool = True,
        max_eigenvalue_lost_retries: int = 50,
    ):
        """Init method.

        Args:
            lammps_runner: a runner whose LAMMPS executable can handle ARTn and the MLIP pair_style.
            initial_configuration: the single starting configuration the ARTn search is launched from.
            artn_input_configuration: the parameters written to artn.in (the &ARTN_PARAMETERS namelist and the
                push); defaults to ArtnInputConfiguration() when None.
            artn_library_plugin_path: path to the compiled ARTn library plugin. When None, it is read from
                the ARTN_PLUGIN_PATH environment variable. A directory is accepted too, in which case
                'libartn-lmp.so' or 'lib/libartn-lmp.so' is looked up inside it.
            number_of_requested_saddles: how many saddles to find (accumulated across the campaign) before the
                run reports SUCCESS; until then each run either finds saddles or is interrupted by uncertainty.
            restart_from_new_min: when True, each search after a saddle restarts from the new minimum found
                across it (a KMC-like hop); when False, every search restarts from the initial configuration.
            max_eigenvalue_lost_retries: how many consecutive 'eigenvalue lost' failures to retry (each with a
                fresh random push) before giving up; the counter resets once a saddle is found or the run is
                interrupted, and starts over at each epoch.
        """
        super().__init__(lammps_runner, initial_configuration)

        self._artn_library_plugin_path = self._resolve_artn_library_plugin_path(artn_library_plugin_path)
        self._artn_input_configuration = artn_input_configuration or ArtnInputConfiguration()
        self._number_of_requested_saddles = number_of_requested_saddles
        self._current_saddle_found = 0
        self._restart_from_new_min = restart_from_new_min
        self._max_eigenvalue_lost_retries = max_eigenvalue_lost_retries

    @staticmethod
    def _resolve_artn_library_plugin_path(artn_library_plugin_path: Optional[Path]) -> Path:
        """Return the plugin path, falling back to the ARTN_PLUGIN_PATH environment variable."""
        if artn_library_plugin_path is None:
            environment_value = os.environ.get(ARTN_PLUGIN_PATH_ENVIRONMENT_VARIABLE)
            if environment_value is None:
                raise ValueError(
                    f"The {ARTN_PLUGIN_PATH_ENVIRONMENT_VARIABLE} environment variable is not defined and "
                    "artn_library_plugin_path was not passed to ArtnDriver. Provide one of the two so the "
                    "compiled ARTn plugin can be located. The ARTn plugin can be obtained from "
                    "https://gitlab.com/mammasmias/artn-plugin."
                )
            artn_library_plugin_path = Path(environment_value)

        if artn_library_plugin_path.is_dir():
            candidate_paths = [
                artn_library_plugin_path / ARTN_LIBRARY_FILE_NAME,
                artn_library_plugin_path / "lib" / ARTN_LIBRARY_FILE_NAME,
            ]
            artn_library_plugin_path = next(
                (candidate for candidate in candidate_paths if candidate.is_file()), artn_library_plugin_path
            )

        assert artn_library_plugin_path.is_file(), "The artn library plugin_path is not valid."
        return artn_library_plugin_path

    def summarize_interruption(self, working_directory: Path) -> List[str]:
        """Build the ARTn interrupted-run log lines from the accumulated per-search summary.

        Sums the ARTn steps and (LAMMPS-step) force evaluations over every search of this run, and takes the
        interrupting search's stage and eigenvalue from the last (interruption) record.
        """
        summaries = read_artn_search_summaries(working_directory)
        total_artn_steps = sum(record["artn_steps"] for record in summaries)
        total_force_evaluations = sum(record["force_evaluations"] or 0 for record in summaries)

        interruption = summaries[-1]
        macro_stage, micro_stage = interruption["macro_stage"], interruption["micro_stage"]
        macro_description = MACRO_STAGE_DESCRIPTIONS.get(macro_stage, macro_stage)
        micro_description = MICRO_STAGE_DESCRIPTIONS.get(micro_stage, micro_stage)
        lines = [
            f"{total_artn_steps} ARTn step, {total_force_evaluations} force evaluation.",
            f"Interrupted in {macro_stage} {micro_stage} ({macro_description}, {micro_description}).",
        ]
        eigenvalue = interruption["lowest_eigenvalue_eV_per_A2"]
        if eigenvalue is not None:  # absent until Lanczos has computed an eigenvalue (early basin steps)
            lines.append(f"lowest_eigval {eigenvalue:.4f} eV/A^2, "
                         f"eigenvector stability a1 {interruption['eigenvector_stability']:.2f}.")
        return lines

    def summarize_success(self, working_directory: Path) -> List[str]:
        """Build the ARTn success log lines: the transition's activation energies, reaction dE and locality."""
        information = collect_artn_transition_information(working_directory)
        return [
            "ARTn found a new transition pathway (1 saddle, 2 minima).",
            f"Forward activation energy {information['forward_activation_energy']:+.3f} eV, "
            f"Backward activation energy {information['backward_activation_energy']:+.3f} eV",
            f"Reaction delta Energy {information['reaction_energy']:+.3f} eV.",
            f"{information['number_of_participating_atoms']} atoms displaced by more than "
            f"{information['displacement_threshold']:g} ang.",
        ]

    def _run_dynamics(self, working_directory: Path, logger: logging.Logger) -> CalculationState:
        """Run ARTn searches until number_of_requested_saddles are found (a running total across the campaign).

        Each found saddle continues the search; an uncertainty halt returns INTERRUPTION, keeping the total so
        the next epoch resumes it; SUCCESS is reported only once the total is reached. An 'eigenvalue lost'
        failure is retried (up to max_eigenvalue_lost_retries) with a fresh random push. Every search appends a
        record to the accumulated summary read back by summarize_interruption.
        """
        logger.info("Launching LAMMPS")
        total_execution_time = 0.0
        calculation_state = CalculationState.SUCCESS
        eigenvalue_lost_retries = 0
        while self._current_saddle_found < self._number_of_requested_saddles:
            self._write_artn_input_file(working_directory)  # fresh push atom for each launch (if not fixed)
            execution_time, succeeded = self._execute_lammps(working_directory, logger)
            total_execution_time += execution_time
            if not succeeded:
                calculation_state = CalculationState.ERROR
                break

            calculation_state = self._get_calculation_state(working_directory)
            if calculation_state == CalculationState.ERROR:
                failure_message = self._read_failure_message(working_directory)
                if (EIGENVALUE_LOST_MESSAGE in (failure_message or "")
                        and eigenvalue_lost_retries < self._max_eigenvalue_lost_retries):
                    eigenvalue_lost_retries += 1
                    logger.info("Eigenvalue lost, retry "
                                f"{eigenvalue_lost_retries}/{self._max_eigenvalue_lost_retries}")
                    continue
                logger.error(f"ARTn search failed: {failure_message or 'no failure message in artn.out'}")
                break

            information = collect_artn_run_information(working_directory)
            if calculation_state == CalculationState.INTERRUPTION:
                append_artn_search_summary(working_directory, dict(
                    outcome="interruption",
                    artn_steps=information["artn_step"],
                    force_evaluations=information["force_evaluations"],
                    macro_stage=information["macro_stage"],
                    micro_stage=information["micro_stage"],
                    lowest_eigenvalue_eV_per_A2=information["lowest_eigenvalue"],
                    eigenvector_stability=information["eigenvector_stability"],
                ))
                break

            eigenvalue_lost_retries = 0
            self._current_saddle_found += 1
            barrier = collect_artn_transition_information(working_directory)["forward_activation_energy"]
            append_artn_search_summary(working_directory, dict(
                outcome="saddle",
                saddle=self._current_saddle_found,
                barrier_eV=barrier,
                artn_steps=information["artn_step"],
                force_evaluations=information["force_evaluations"],
            ))
            logger.info(f"Found saddle {self._current_saddle_found}/{self._number_of_requested_saddles}, "
                        f"barrier {barrier:.3f} eV")
            if self._restart_from_new_min:
                self._hop_to_new_minimum(working_directory)
        logger.info(f"Total execution time: {total_execution_time: 6.3e} sec.")
        return calculation_state

    def _hop_to_new_minimum(self, working_directory: Path) -> None:
        """Restart the next search from the new minimum across the saddle (the min that is not where we started).

        ARTn relaxes to two minima per saddle (min1/min2) in no particular order; the one equivalent to the
        current configuration is the basin we came from, so the other is the new minimum to hop to.
        """
        specorder = generate_named_elements_blocks(self.initial_configuration)[2].split()
        first_minimum = read_artn_xyz(working_directory / "min1.xyz", specorder)
        second_minimum = read_artn_xyz(working_directory / "min2.xyz", specorder)
        position_tolerance = self._artn_input_configuration.delr_thr
        new_minimum = (
            second_minimum
            if configurations_are_equivalent(first_minimum, self._current_configuration, position_tolerance)
            else first_minimum
        )
        self._write_configuration(new_minimum, working_directory)
        self._current_configuration = new_minimum

    def _prepare_reference_files(self, working_directory: Path) -> None:
        """Write the ARTn input file (artn.in) into the working directory."""
        self._write_artn_input_file(working_directory)

    def _write_artn_input_file(self, working_directory: Path) -> None:
        """Write artn.in, resolving the pushed atom (a random atom each launch when push_ids is None)."""
        configuration = replace(self._artn_input_configuration, push_ids=self._resolve_push_ids())
        write_artn_input_file(working_directory / "artn.in", configuration)

    def _resolve_push_ids(self) -> int:
        """The pushed atom: the fixed push_ids if given, else a random atom of the current configuration."""
        if self._artn_input_configuration.push_ids is not None:
            return self._artn_input_configuration.push_ids
        return random.randint(1, len(self._current_configuration))

    def _dynamics_block(self) -> str:
        """Build the ARTn dynamics commands (plugin load + fix artn + FIRE minimization)."""
        return build_artn_lammps_tail(self._artn_library_plugin_path)

    def _trajectory_dump_block(self, elements_string: str, dump_fields: str) -> str:
        """Skip the full-trajectory dump for ARTn (it can grow huge over many searches)."""
        return ""

    def _get_calculation_state(self, working_directory: Path) -> CalculationState:
        """Parse artn.out for the ARTn outcome (ERROR if the file is missing)."""
        artn_output_file_path = working_directory / "artn.out"
        if not artn_output_file_path.is_file():
            return CalculationState.ERROR
        with open(artn_output_file_path, "r") as file_descriptor:
            return get_calculation_state_from_artn_output(file_descriptor.read())

    def _read_failure_message(self, working_directory: Path) -> Optional[str]:
        """The ARTn 'Failure message:' line from artn.out, or None when the file or the line is absent."""
        artn_output_file_path = working_directory / "artn.out"
        if not artn_output_file_path.is_file():
            return None
        return parse_artn_failure_message(artn_output_file_path.read_text())

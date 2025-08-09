import logging
import shutil
import time
from pathlib import Path
from string import Template
from typing import Optional

from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.artn.artn_outputs import \
    get_calculation_state_from_artn_output
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.artn.calculation_state import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.dynamic_driver import \
    PATH_TO_LAMMPS_ARTN_TEMPLATE
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.inputs import \
    generate_named_elements_blocks
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.lammps_runner import \
    LammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    configure_logging


class ArtnDriver:
    """ARTn Driver.

    This class is responsible for driving the execution of a ARTn simulation with LAMMPS.
    """

    def __init__(self, lammps_runner: LammpsRunner, artn_library_plugin_path: Path, reference_directory: Path,
                 template_path: Optional[Path] = None, mtp_potential_path: Optional[Path] = None,):
        """Init method.

        Args:
            lammps_runner: a class that can drive the execution of LAMMPS. It is assumed that
                the underlying LAMMPS executable can properly handle ARTn and mapped FLARE models.
            artn_library_plugin_path: Path to the compiled artn library plugin. It is assumed to be compatible
                with the lammps runner.
            reference_directory: Path to a directory that is assumed to contain a file 'artn.in' defining
                the ARtn job to run, and 'initial_configuration.dat' defining the starting configuration.
            template_path: optional LAMMPS input template to use instead of PATH_TO_LAMMPS_ARTN_TEMPLATE.
            mtp_potential_path: optional path to MTP potential file (used by MTP template).
        """
        assert reference_directory.is_dir(), "The reference directory is not valid."

        assert artn_library_plugin_path.is_file(), "The artn library plugin_path is not valid."
        self._artn_library_plugin_path = artn_library_plugin_path

        self._reference_artn_in_file_path = reference_directory / "artn.in"
        assert self._reference_artn_in_file_path.is_file(), "The reference artn.in file does not exist."

        self._initial_configuration_file_path = reference_directory / "initial_configuration.dat"
        assert self._initial_configuration_file_path.is_file(), "The initial configuration file does not exist."
        self.initial_structure = self._load_initial_configuration(self._initial_configuration_file_path)

        self._lammps_runner = lammps_runner

        # load the template (prefer user-provided path if given)
        template_file = (template_path if template_path is not None else PATH_TO_LAMMPS_ARTN_TEMPLATE)
        with open(template_file, mode="r") as fd:
            template_string = fd.read()
        self._template = Template(template_string)
        # optional: carried through to template as $mtp_potential_path
        self._mtp_potential_path = mtp_potential_path

        self._lammps_input_filename = "lammps.in"

    @staticmethod
    def _load_initial_configuration(initial_configuration_file_path: Path) -> Structure:
        """Load the initial configuration as a structure."""
        try:
            structure = LammpsData.from_file(str(initial_configuration_file_path),
                                             atom_style="atomic",
                                             sort_id=True).structure
        except Exception:
            raise ValueError(f"The initial configuration file {initial_configuration_file_path} cannot be loaded.\n"
                             "Make sure the file is present and in a format that can be read by the "
                             "LammpsData module.")

        return structure

    def run(self, working_directory: Path, uncertainty_threshold: float,
            pair_coeff_file_path: Optional[Path], mapped_uncertainty_file_path: Optional[Path]) -> CalculationState:
        """Run the ARTn simulation.

        Args:
            working_directory: Path to the working directory when the run will be performed.
            uncertainty_threshold: uncertainty threshold for stopping simulation.
            pair_coeff_file_path: path to the mapped FLARE coefficients.
            mapped_uncertainty_file_path: path to the mapped uncertainty FLARE coefficients.

        Returns:
            calculation_state: status of the ARTn calculation.
        """
        assert not working_directory.is_dir(), \
            f"The working directory {working_directory} already exists! Exiting to avoid writing over existing data."

        working_directory.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger('artn_run')
        configure_logging(experiment_dir=str(working_directory),
                          logger=logger,
                          log_to_console=False)

        logger.info("Copying the reference artn.in file to the working directory.")
        shutil.copy(self._reference_artn_in_file_path, str(working_directory / "artn.in"))

        logger.info("Write the starting configuration to the working directory.")
        lammps_data = LammpsData.from_structure(self.initial_structure, atom_style="atomic")
        lammps_data.write_file(str(working_directory / "initial_configuration.dat"))

        logger.info("Write the LAMMPS input script, with parameters:")
        logger.info(f"   - uncertainty_threshold = {uncertainty_threshold}")
        logger.info(f"   - pair_coeff_file_path = {pair_coeff_file_path}")
        logger.info(f"   - mapped_uncertainty_file_path = {mapped_uncertainty_file_path}")

        group_block, mass_block, elements_string = generate_named_elements_blocks(self.initial_structure)

        parameters = dict(configuration_file_path="initial_configuration.dat",
                          pair_coeff_file_path = (
                              "" if pair_coeff_file_path is None else str(pair_coeff_file_path)
                          ),
                          mapped_uncertainty_file_path = (
                             "" if mapped_uncertainty_file_path is None else str(mapped_uncertainty_file_path)
                          ),
                          artn_library_plugin_path=str(self._artn_library_plugin_path),
                          uncertainty_threshold=f"{uncertainty_threshold:.12f}",
                          group_block=group_block,
                          mass_block=mass_block,
                          elements_string=elements_string,
                          mtp_potential_path=(
                              "" if self._mtp_potential_path is None else str(self._mtp_potential_path)
                          ))

        script_content = self._template.safe_substitute(**parameters)
        input_file_path = working_directory / self._lammps_input_filename
        with open(input_file_path, 'w') as fd:
            fd.write(script_content)

        logger.info("Launching LAMMPS")
        time1 = time.time()
        self._lammps_runner.run_lammps(working_directory=working_directory,
                                       lammps_input_file_name=self._lammps_input_filename)
        time2 = time.time()
        logger.info(f"LAMMPS execution has finished. Execution Time: {time2-time1: 6.3e} sec.")

        artn_output_file_path = working_directory / "artn.out"
        assert artn_output_file_path.is_file(), "The artn output file, 'artn.out', is missing. Something went wrong."

        with open(artn_output_file_path, "r") as fd:
            artn_output = fd.read()

        del logger
        return get_calculation_state_from_artn_output(artn_output)

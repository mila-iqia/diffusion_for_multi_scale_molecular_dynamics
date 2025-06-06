import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Union

from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.inputs import LammpsTemplateGen

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps import \
    PATH_TO_SINGLE_POINT_CALCULATION_TEMPLATE
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.inputs import \
    generate_named_elements_blocks
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.lammps_runner import \
    LammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.outputs import \
    extract_all_fields_from_dump
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_single_point_calculator import (  # noqa
    BaseSinglePointCalculator, SinglePointCalculation)


class BaseLAMMPSSinglePointCalculator(BaseSinglePointCalculator):
    """Base LAMMPS Single Point Calculator.

    This base class implements generic interactions with LAMMPS, which is assumed to
    be available through an executable.

    Interacting with LAMMPS rely on the pymatgen.io.lammps module to transform Structure
    objects into a LAMMPS readable file and to drive the generation of the LAMMPS input script.

    The specific potential to be used must be implemented in a child class.
    """

    @abstractmethod
    def _generate_pair_coeff_command(self, elements_string: str) -> str:
        raise NotImplementedError("must be implemented in child class.")

    @abstractmethod
    def _generate_pair_style_command(self) -> str:
        raise NotImplementedError("must be implemented in child class.")

    @abstractmethod
    def _generate_uncertainty_variable_string(self) -> str:
        raise NotImplementedError("must be implemented in child class.")

    def __init__(self, lammps_runner: LammpsRunner, **kwargs):
        """Init method."""
        super().__init__(self)
        self._calculation_type = "LAMMPS"
        self._lammps_runner = lammps_runner

        self._input_file_name = "lammps.in"
        self._data_filename = "configuration.dat"

    def _extract_calculation_results(
        self, working_directory: str
    ) -> SinglePointCalculation:
        lammps_dump_path = Path(working_directory) / "dump.yaml"

        list_structures, list_forces, list_energies, list_uncertainties = (
            extract_all_fields_from_dump(lammps_dump_path)
        )
        assert (
            len(list_structures) == 1
        ), "There is more than one frame in the dump file. This is not 'single point'!"

        result = SinglePointCalculation(
            calculation_type=self._calculation_type,
            structure=list_structures[0],
            forces=list_forces[0],
            energy=list_energies[0],
            uncertainties=list_uncertainties[0],
        )

        return result

    def _generate_settings_dictionary(self, structure: Structure) -> Dict:
        """Generate the settings dictionary needed by Pymatgen's templating method."""
        group_block, mass_block, elements_string = generate_named_elements_blocks(structure)

        settings = dict(
            configuration_file_path=self._data_filename,
            pair_style_command=self._generate_pair_style_command(),
            pair_coeff_command=self._generate_pair_coeff_command(elements_string),
            uncertainty_variable_name=self._generate_uncertainty_variable_string(),
            group_block=group_block,
            mass_block=mass_block,
            elements_string=elements_string,
        )

        return settings

    def calculate_in_work_directory(
        self, structure: Structure, work_directory: Union[Path, str]
    ) -> SinglePointCalculation:
        """Calculate in work directory.

        Drive LAMMPS execution in a given working directory.

        Args:
            structure: pymatgen structure.
            work_directory: work directory where inputs and outputs will be recorded..

        Returns:
            calculation_results: the parsed LAMMPS output.
        """
        Path(work_directory).mkdir(parents=True, exist_ok=True)

        settings = self._generate_settings_dictionary(structure)
        lammps_data = LammpsData.from_structure(structure, atom_style="atomic")

        input_set = LammpsTemplateGen().get_input_set(
            script_template=PATH_TO_SINGLE_POINT_CALCULATION_TEMPLATE,
            settings=settings,
            script_filename=self._input_file_name,
            data=lammps_data,
            data_filename=self._data_filename,
        )

        input_set.write_input(work_directory)

        self._lammps_runner.run_lammps(working_directory=work_directory,
                                       lammps_input_file_name=self._input_file_name)

        calculation_result = self._extract_calculation_results(work_directory)

        return calculation_result

    def calculate(self, structure: Structure) -> SinglePointCalculation:
        """Calculate.

        Drive LAMMPS execution.

        Args:
            structure: pymatgen structure.

        Returns:
            calculation_results: the parsed LAMMPS output.
        """
        with tempfile.TemporaryDirectory() as tmp_work_dir:
            calculation_result = self.calculate_in_work_directory(
                structure, tmp_work_dir
            )

        return calculation_result

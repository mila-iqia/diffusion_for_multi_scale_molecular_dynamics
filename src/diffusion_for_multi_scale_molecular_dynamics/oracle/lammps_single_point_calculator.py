import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from ase import Atoms
from ase.io.lammpsdata import write_lammps_data

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.inputs import \
    sort_atoms_elements_by_atomic_mass
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.outputs import \
    extract_all_fields
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.single_point_calc_lammps_input import (
    LammpsInputBuilder, write_lammps_input)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    CONFIGURATION_FILENAME, DUMP_FILENAME, ENERGY_FILENAME,
    LAMMPS_INPUT_FILENAME, numbered_filename)
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import (  # noqa
    BaseSinglePointCalculator, SinglePointCalculation)
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)


class LammpsSinglePointCalculator(BaseSinglePointCalculator):
    """LAMMPS Single Point Calculator.

    Drive a single-point LAMMPS calculation for a given potential through an injected runner.
    """

    def __init__(
        self,
        lammps_potential: LammpsPotential,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
        with_uncertainty: bool = False,
    ):
        """Init method.

        Args:
            lammps_potential: the potential to evaluate (emits the pair commands).
            lammps_runner: a runner that executes LAMMPS (subprocess or in-process).
            with_uncertainty: whether to compute the per-atom uncertainty.
        """
        super().__init__(self)
        self._potential = lammps_potential
        self._lammps_runner = lammps_runner
        self._with_uncertainty = with_uncertainty
        self._input_builder = LammpsInputBuilder()

        self._calculation_type = lammps_potential.calculation_type
        self.name = lammps_potential.name  # the oracle's log name is the potential's (e.g. 'SW')
        self._input_file_name = LAMMPS_INPUT_FILENAME
        self._data_filename = CONFIGURATION_FILENAME

    def _extract_calculation_results(
        self, working_directory: str, dump_filename: str = DUMP_FILENAME, energy_filename: str = ENERGY_FILENAME
    ) -> SinglePointCalculation:
        lammps_dump_path = Path(working_directory) / dump_filename

        list_atoms, list_forces, list_uncertainties = (
            extract_all_fields(lammps_dump_path, uncertainty_field=self._potential.uncertainty_field())
        )
        assert (
            len(list_atoms) == 1
        ), "There is more than one frame in the dump file. This is not 'single point'!"

        energy = float((Path(working_directory) / energy_filename).read_text().strip())

        result = SinglePointCalculation(
            calculation_type=self._calculation_type,
            atoms=list_atoms[0],
            forces=list_forces[0],
            energy=energy,
            uncertainties=list_uncertainties[0],
        )

        return result

    @staticmethod
    def _write_lammps_data(atoms: Atoms, path: Path) -> None:
        """Write an ase.Atoms to a LAMMPS data file, with a mass-sorted species order."""
        specorder = [symbol for symbol, _ in sort_atoms_elements_by_atomic_mass(atoms)]
        write_lammps_data(str(path), atoms, atom_style="atomic", specorder=specorder)

    def _build_input(self, atoms: Atoms) -> str:
        """Build the LAMMPS input script for a single-point calculation."""
        return self._input_builder.build_single_point(
            atoms,
            self._potential,
            with_uncertainty=self._with_uncertainty,
            configuration_filename=self._data_filename,
        )

    def calculate_in_work_directory(
        self, atoms: Atoms, work_directory: Union[Path, str]
    ) -> SinglePointCalculation:
        """Calculate in work directory.

        Drive LAMMPS execution in a given working directory.

        Args:
            atoms: the configuration to evaluate.
            work_directory: work directory where inputs and outputs will be recorded.

        Returns:
            calculation_results: the parsed LAMMPS output.
        """
        work_directory = Path(work_directory)
        work_directory.mkdir(parents=True, exist_ok=True)

        self._write_lammps_data(atoms, work_directory / self._data_filename)

        input_content = self._build_input(atoms)
        write_lammps_input(input_content, work_directory / self._input_file_name)

        self._lammps_runner.run_lammps(working_directory=work_directory,
                                       lammps_input_file_name=self._input_file_name)

        return self._extract_calculation_results(str(work_directory))

    def calculate(self, atoms: Atoms, results_path: Optional[Path] = None) -> SinglePointCalculation:
        """Calculate.

        Drive LAMMPS execution.

        Args:
            atoms: the configuration to evaluate.
            results_path: (Optional) if present, the text dump file produced by the LAMMPS calculation will
                be moved to this location.

        Returns:
            calculation_results: the parsed LAMMPS output.
        """
        with tempfile.TemporaryDirectory() as tmp_work_dir:
            calculation_result = self.calculate_in_work_directory(atoms, tmp_work_dir)
            if results_path is not None:
                src = os.path.join(tmp_work_dir, DUMP_FILENAME)
                dst = str(results_path)
                shutil.move(src, dst)

        return calculation_result

    def calculate_many_in_work_directory(
        self, list_atoms: List[Atoms], work_directory: Union[Path, str]
    ) -> List[SinglePointCalculation]:
        """Evaluate several configurations with a single LAMMPS run in a given working directory.

        A single looping input reads one data file and writes one dump file per configuration, so LAMMPS is
        launched only once (one process for the subprocess runner, one instance for the in-process runner).

        Args:
            list_atoms: the configurations to evaluate, in order.
            work_directory: work directory where inputs and outputs will be recorded.

        Returns:
            calculation_results: the parsed LAMMPS output, one per configuration (in the same order).
        """
        work_directory = Path(work_directory)
        work_directory.mkdir(parents=True, exist_ok=True)

        configuration_filenames = [numbered_filename(CONFIGURATION_FILENAME, index) for index in range(len(list_atoms))]
        dump_filenames = [numbered_filename(DUMP_FILENAME, index) for index in range(len(list_atoms))]
        energy_filenames = [numbered_filename(ENERGY_FILENAME, index) for index in range(len(list_atoms))]

        for atoms, configuration_filename in zip(list_atoms, configuration_filenames):
            self._write_lammps_data(atoms, work_directory / configuration_filename)

        input_content = self._input_builder.build_looping_single_point(
            list_atoms, self._potential, configuration_filenames, dump_filenames, energy_filenames,
            with_uncertainty=self._with_uncertainty,
        )
        write_lammps_input(input_content, work_directory / self._input_file_name)

        self._lammps_runner.run_lammps(working_directory=work_directory,
                                       lammps_input_file_name=self._input_file_name)

        return [self._extract_calculation_results(str(work_directory), dump_filename, energy_filename)
                for dump_filename, energy_filename in zip(dump_filenames, energy_filenames)]

    def calculate_many(self, list_atoms: List[Atoms]) -> List[SinglePointCalculation]:
        """Evaluate several configurations with a single LAMMPS run (see ``calculate_many_in_work_directory``)."""
        if not list_atoms:
            return []
        with tempfile.TemporaryDirectory() as tmp_work_dir:
            return self.calculate_many_in_work_directory(list_atoms, tmp_work_dir)

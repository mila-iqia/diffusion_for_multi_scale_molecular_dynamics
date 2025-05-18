import subprocess
import tempfile
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import yaml
from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp.calculator import SGP_Calculator

import numpy as np
from ase import Atoms
from yaml import CLoader

from flare_experiment.utilities.utils import parse_lammps_dump


@dataclass(kw_only=True)
class Element:
    """A simple data structure to describe all we need for the elements"""
    name: str
    mass: float
    atomic_number: int

# We can find a better solution later.
ELEMENTS_DICT = {14: Element(name='Si', mass=28.0855, atomic_number=14),
                 32: Element(name='Ge', mass=72.64, atomic_number=32)}

@dataclass(kw_only=True)
class CalculationResult:
    """A structure and corresponding labels, for training a sparse GP."""
    calculation_type: str
    atoms: Atoms
    forces: np.ndarray
    energy: float
    force_uncertainties: Optional[np.ndarray] = None
    # Local energies. FLARE's code is cryptic, so it is hard to know exactly what is being computed.
    energy_uncertainties: Optional[np.ndarray] = None
    spatial_dimension: int = 3


class SinglePointCalculator:
    def __init__(self, args, **kwargs):
        pass

    @abstractmethod
    def calculate(self, atoms: Atoms) -> CalculationResult:
        """This method just defines the API."""
        raise NotImplementedError("This mmethod must be implemented in a base class.")


class FlareSinglePointCalculator(SinglePointCalculator):
    """Wrap around the horrible flare calculator class."""

    def __init__(self, sgp_model: SGP_Wrapper):
        super().__init__(self)
        self._calculation_type = "flare_sgp"
        self._flare_calculator = SGP_Calculator(sgp_model)
        self._calculation_properties = ["energy", "forces", "stds"]

        self._uncertainty_is_energy = None

        match sgp_model.variance_type:
            case "local":
                self._uncertainty_is_energy = True
            case "DTC":
                self._uncertainty_is_energy = False
            case _:
                raise NotImplementedError("Only local and DTC variance types are implemented. Review input.")

    def calculate(self, atoms: Atoms) -> CalculationResult:

        self._flare_calculator.calculate(atoms=atoms, properties=self._calculation_properties)

        energy = self._flare_calculator.results["energy"]
        forces = self._flare_calculator.results["forces"]

        # FLARE's code is cryptic, so it is hard to know exactly what is being computed.
        # Scanning the code flare.bffs.sgp.calculator.SGP_Calculator.predict_on_structure,
        # it seems that the 'stds' array is of the same dimensions as 'forces'. It contains
        # force uncertainty if variance_type = 'DTC', and local energy uncertainty if variance_type = 'local',
        # shoved in the first column.
        flare_stds = self._flare_calculator.results["stds"]

        if self._uncertainty_is_energy:
            # FLARE's code normalizes this to sigma internally. The energy uncertainty is unitless.
            energy_uncertainties = flare_stds[:, 0]
            force_uncertainties = None
        else:
            energy_uncertainties = None
            force_uncertainties = flare_stds

        return CalculationResult(calculation_type=self._calculation_type,
                                 atoms=atoms,
                                 energy=energy,
                                 forces=forces,
                                 energy_uncertainties=energy_uncertainties,
                                 force_uncertainties=force_uncertainties)


class LAMMPSSinglePointCalculator(SinglePointCalculator):

    def __init__(self, lammps_executable_path: str, **kwargs):
        super().__init__(self)
        self._lammps_executable_path = lammps_executable_path

        self._input_file_name = "in.lammps"
        # Extra flags to keep LAMMPS quiet.
        self._commands = [f"{self._lammps_executable_path}",
                          "-log",
                          "none",
                          "-echo",
                          "none",
                          "-screen",
                          "none",
                          "-i",
                          self._input_file_name]

    def _get_dump_document(self, working_directory: str) -> Dict:
        with open(Path(working_directory) / "dump.yaml", "r") as fd:
            dump_yaml = yaml.load_all(fd, Loader=CLoader)
            list_docs = list(dump_yaml)
            assert len(list_docs) == 1, "There is more than one frame in the dump file. This is not 'single point'!"

            dump_doc = list_docs[0]

        return dump_doc

    @abstractmethod
    def _generate_pair_style_commands(self, elements_string: str) -> List[str]:
        raise NotImplementedError("must be implemented in child class.")

    @abstractmethod
    def _generate_dump_commands(self, elements_string: str) -> List[str]:
        raise NotImplementedError("must be implemented in child class.")

    def _generate_lammps_script(self, atoms: Atoms) -> str:

        atoms.wrap() # Make sure the positions are within the periodic unit cell.

        a1, a2, a3 = np.diag(atoms.get_cell())
        atomic_numbers = atoms.get_atomic_numbers()
        cartesian_positions = atoms.positions
        unique_atomic_numbers = np.sort(np.unique(atomic_numbers))

        number_of_atom_types = len(unique_atomic_numbers)

        commands = []
        commands.append("units metal")
        commands.append("atom_style atomic")
        commands.append(f"region simbox block 0 {a1} 0 {a2} 0 {a3}")
        commands.append(f"create_box {number_of_atom_types} simbox")

        elements_string = ""
        # don't start the groups at zero
        group_id_dictionary = dict()
        for group_id, atomic_number in enumerate(unique_atomic_numbers, 1):
            group_id_dictionary[atomic_number] = group_id
            element = ELEMENTS_DICT[atomic_number]
            elements_string += f" {element.name}"
            commands.append(f"group {element.name} type {group_id}")
            commands.append(f"mass {group_id} {element.mass}")

        pair_style_commands = self._generate_pair_style_commands(elements_string)
        commands.extend(pair_style_commands)

        for atomic_number, cartesian_position in zip(atomic_numbers, cartesian_positions):
            group_id = group_id_dictionary[atomic_number]
            positions_string = " ".join(map(str, cartesian_position))
            commands.append(f"create_atoms {group_id} single {positions_string}")

        commands.append("fix 1 all nvt temp 300 300 0.01")

        dump_commands = self._generate_dump_commands(elements_string)

        commands.extend(dump_commands)

        commands.append("run 0")

        lammps_script_content = "\n".join(commands)
        return lammps_script_content


    @abstractmethod
    def _extract_calculation_results(self, parsed_results: Dict) -> CalculationResult:
        raise NotImplementedError("Must be implemented in child class.")

    def _parse_output(self, tmp_work_dir: str)-> CalculationResult:
        parsed_output = parse_lammps_dump(lammps_dump=str(Path(tmp_work_dir) / "dump.yaml"))
        calculation_results = self._extract_calculation_results(parsed_output)

        return calculation_results

    def calculate(self, atoms: Atoms) -> CalculationResult:

        lammps_script_content = self._generate_lammps_script(atoms)

        with tempfile.TemporaryDirectory() as tmp_work_dir:
            with open(Path(tmp_work_dir) / self._input_file_name, "w") as fd:
                fd.write(lammps_script_content)

            subprocess.run(
                self._commands,
                cwd=tmp_work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,  # Decode stdout and stderr as text
                check=True  # Raise a CalledProcessError for non-zero exit codes
            )

            calculation_result = self._parse_output(tmp_work_dir)

        return calculation_result


class StillingerWeberSinglePointCalculator(LAMMPSSinglePointCalculator):

    def __init__(self, lammps_executable_path: str, sw_coefficients_file_path: str):
        super().__init__(lammps_executable_path)

        self._calculation_type = "stillinger_weber"
        self._sw_coefficients_file_path = sw_coefficients_file_path

    def _generate_pair_style_commands(self, elements_string: str) -> List[str]:
        commands = []
        commands.append("pair_style sw")
        commands.append(f"pair_coeff * * {self._sw_coefficients_file_path}{elements_string}")
        return commands

    def _generate_dump_commands(self, elements_string: str) -> List[str]:
        commands = []
        commands.append(f"dump 1 all yaml 1 dump.yaml id element x y z fx fy fz")
        commands.append(f"dump_modify 1 element {elements_string}")
        commands.append("dump_modify 1 thermo yes")
        commands.append("thermo 1")
        commands.append("thermo_style custom pe")

        return commands

    def _extract_calculation_results(self, parsed_results: Dict) -> CalculationResult:
        calculation_results = CalculationResult(calculation_type=self._calculation_type,
                                                atoms=parsed_results['atoms'][0],
                                                forces=parsed_results['forces'][0],
                                                energy=parsed_results['energy'][0])
        return calculation_results



class MappedFlareSinglePointCalculator(LAMMPSSinglePointCalculator):

    def __init__(self, lammps_executable_path: str, pair_coeff_path: str, map_unc_path: str):
        super().__init__(lammps_executable_path)

        self._calculation_type = "mapped_flare"
        self._pair_coeff_path = pair_coeff_path
        self._map_unc_path = map_unc_path

    def _generate_pair_style_commands(self, elements_string: str) -> List[str]:
        commands = []
        commands.append("pair_style flare")
        commands.append(f"pair_coeff * * {self._pair_coeff_path}")
        commands.append(f"compute unc all flare/std/atom {self._map_unc_path}")
        return commands

    def _generate_dump_commands(self, elements_string: str) -> List[str]:
        commands = []
        commands.append(f"dump 1 all yaml 1 dump.yaml id element x y z fx fy fz c_unc")
        commands.append(f"dump_modify 1 element {elements_string}")
        commands.append("dump_modify 1 thermo yes")
        commands.append("thermo 1")
        commands.append("thermo_style custom pe")
        return commands

    def _extract_calculation_results(self, parsed_results: Dict) -> CalculationResult:
        calculation_results = CalculationResult(calculation_type=self._calculation_type,
                                                atoms=parsed_results['atoms'][0],
                                                forces=parsed_results['forces'][0],
                                                energy=parsed_results['energy'][0],
                                                energy_uncertainties=parsed_results['uncertainties'][0])
        return calculation_results

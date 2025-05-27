from abc import abstractmethod
from pathlib import Path
from typing import Dict, List

import yaml
from pymatgen.core import Structure
from yaml import CLoader

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_single_point_calculator import \
    BaseSinglePointCalculator  # noqa


class BaseLAMMPSSinglePointCalculator(BaseSinglePointCalculator):
    """Base LAMMPS Single Point Calculator.

    This base class implements generic interactions with LAMMPS, which is assumed to
    be available through an executable.

    The specific potential to be used must be implemented in a child class.
    """

    @abstractmethod
    def _generate_pair_style_commands(self, elements_string: str) -> List[str]:
        raise NotImplementedError("must be implemented in child class.")

    @abstractmethod
    def _generate_dump_commands(self, elements_string: str) -> List[str]:
        raise NotImplementedError("must be implemented in child class.")

    def __init__(self, lammps_executable_path: Path, **kwargs):
        """Init method."""
        super().__init__(self)
        assert (
            lammps_executable_path.is_file()
        ), f"The path {lammps_executable_path} does not exist."
        self._lammps_executable_path = lammps_executable_path

        self._input_file_name = "in.lammps"

        # Extra flags to keep LAMMPS quiet.
        self._commands = [
            f"{self._lammps_executable_path}",
            "-log",
            "none",
            "-echo",
            "none",
            "-screen",
            "none",
            "-i",
            self._input_file_name,
        ]

    def _get_dump_document(self, working_directory: str) -> Dict:
        with open(Path(working_directory) / "dump.yaml", "r") as fd:
            dump_yaml = yaml.load_all(fd, Loader=CLoader)
            list_docs = list(dump_yaml)
            assert (
                len(list_docs) == 1
            ), "There is more than one frame in the dump file. This is not 'single point'!"

            dump_doc = list_docs[0]

        return dump_doc

    def _generate_lammps_script(self, structure: Structure) -> str:
        """Generate LAMMPS script."""
        a1, a2, a3 = structure.lattice.abc
        cartesian_positions = structure.cart_coords
        all_atomic_numbers = [site.specie.Z for site in structure.sites]

        group_id_map = dict()
        mass_map = dict()
        symbol_map = dict()

        list_unique_atomic_numbers = []
        for group_id, element in enumerate(structure.elements, 1):
            atomic_number = element.Z
            list_unique_atomic_numbers.append(atomic_number)
            group_id_map[atomic_number] = group_id
            mass_map[atomic_number] = element.atomic_mass
            symbol_map[atomic_number] = element.symbol

        number_of_atom_types = len(list_unique_atomic_numbers)

        commands = []
        commands.append("units metal")
        commands.append("atom_style atomic")
        commands.append(f"region simbox block 0 {a1} 0 {a2} 0 {a3}")
        commands.append(f"create_box {number_of_atom_types} simbox")

        elements_string = ""
        for atomic_number in list_unique_atomic_numbers:
            group_id = group_id_map[atomic_number]
            symbol = symbol_map[atomic_number]
            mass = mass_map[atomic_number]

            elements_string += f" {symbol}"
            commands.append(f"group {symbol} type {group_id}")
            commands.append(f"mass {group_id} {mass}")

        pair_style_commands = self._generate_pair_style_commands(elements_string)
        commands.extend(pair_style_commands)

        for atomic_number, cartesian_position in zip(
            all_atomic_numbers, cartesian_positions
        ):
            group_id = group_id_map[atomic_number]
            positions_string = " ".join(map(str, cartesian_position))
            commands.append(f"create_atoms {group_id} single {positions_string}")

        commands.append("fix 1 all nvt temp 300 300 0.01")

        dump_commands = self._generate_dump_commands(elements_string)

        commands.extend(dump_commands)

        commands.append("run 0")

        lammps_script_content = "\n".join(commands)
        return lammps_script_content

"""Assemble a LAMMPS input script from composable blocks."""

from pathlib import Path
from typing import List, Union

from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.inputs import \
    generate_named_elements_blocks
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    CONFIGURATION_FILENAME, DUMP_FILENAME, ENERGY_FILENAME)


class LammpsInputBuilder:
    """Assemble a LAMMPS input script from six ordered blocks.

    The blocks are: initialization, configuration, potential, interruption, output and run.
    Some blocks can be empty depending on the calculations. run block could be run 0 by default.
    The build function assemble them together to make the content of the file and the
    write_lammps_input writes it to a file.
    """

    def build_single_point(
        self,
        structure: Structure,
        potential: LammpsPotential,
        with_uncertainty: bool = False,
        configuration_filename: str = CONFIGURATION_FILENAME,
        dump_filename: str = DUMP_FILENAME,
        energy_filename: str = ENERGY_FILENAME,
    ) -> str:
        """Assemble the LAMMPS input for a single-point calculation.

        The structure, forces and species are written to a LAMMPS text dump (read back with ase); the total
        potential energy is written on its own to ``energy_filename``.

        Args:
            structure: the configuration to evaluate.
            potential: the potential providing the interaction commands and dump fields.
            with_uncertainty: whether to compute the per-atom uncertainty.
            configuration_filename: name of the data file read by 'read_data'.
            dump_filename: name of the text dump file the per-atom results are written to.
            energy_filename: name of the file the total potential energy is written to.

        Returns:
            input_script: the assembled LAMMPS input script.
        """
        group_block, mass_block, elements_string = generate_named_elements_blocks(structure)

        blocks = [
            self._initialization_block(),
            self._configuration_block(configuration_filename, group_block, mass_block),
            "\n".join(potential.interaction_commands(elements_string, with_uncertainty=with_uncertainty)),
            self._output_block(elements_string, potential.dump_fields(with_uncertainty=with_uncertainty),
                               dump_filename),
            "run 0",
            self._energy_block(energy_filename),
        ]

        return "\n\n\n".join(block.strip() for block in blocks if block and block.strip()) + "\n"

    def build_looping_single_point(
        self,
        structures: List[Structure],
        potential: LammpsPotential,
        configuration_filenames: List[str],
        dump_filenames: List[str],
        energy_filenames: List[str],
        with_uncertainty: bool = False,
    ) -> str:
        """Assemble a single LAMMPS input that evaluates several configurations in one run.

        Each configuration is a self-contained single-point block (reading its own data file and writing its
        own dump and energy file); consecutive blocks are separated by 'clear', which resets LAMMPS between
        configurations.

        Args:
            structures: the configurations to evaluate, in order.
            potential: the potential providing the interaction commands and dump fields.
            configuration_filenames: the data file name read for each structure (same length as structures).
            dump_filenames: the text dump file written for each structure (same length as structures).
            energy_filenames: the energy file written for each structure (same length as structures).
            with_uncertainty: whether to compute the per-atom uncertainty.

        Returns:
            input_script: the assembled LAMMPS input script.
        """
        blocks = [
            self.build_single_point(structure, potential, with_uncertainty=with_uncertainty,
                                    configuration_filename=configuration_filename, dump_filename=dump_filename,
                                    energy_filename=energy_filename)
            for structure, configuration_filename, dump_filename, energy_filename in zip(
                structures, configuration_filenames, dump_filenames, energy_filenames
            )
        ]
        return "\n\nclear\n\n".join(block.strip() for block in blocks) + "\n"

    def _initialization_block(self) -> str:
        return "units metal\natom_style atomic"

    def _configuration_block(self, configuration_filename: str, group_block: str, mass_block: str) -> str:
        return f"read_data {configuration_filename}\n{group_block.strip()}\n\n{mass_block.strip()}"

    def _output_block(self, elements_string: str, dump_fields: List[str], dump_filename: str = DUMP_FILENAME) -> str:
        fields = " ".join(dump_fields)
        return (
            f"dump dump_id all custom 1 {dump_filename} {fields}\n"
            f"dump_modify dump_id element {elements_string}\n"
            "dump_modify dump_id sort id"
        )

    def _energy_block(self, energy_filename: str = ENERGY_FILENAME) -> str:
        # Write just the total potential energy; the text dump above carries the per-atom data.
        return f'print "$(pe:%.16g)" file {energy_filename} screen no'


def write_lammps_input(content: str, input_file_path: Union[Path, str]) -> None:
    """Write an assembled LAMMPS input script to disk."""
    with open(str(input_file_path), "w") as file_descriptor:
        file_descriptor.write(content)

"""Call LAMMPS to get the forces and energy in a given configuration."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import lammps
import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Element

from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.oracle import \
    SW_COEFFICIENTS_DIR


@dataclass(kw_only=True)
class LammpsOracleParameters:
    """Lammps Oracle Parameters."""
    sw_coeff_filename: str  # Stillinger-Weber potential filename


class LammpsCalculator:
    """Lammps calculator.

    This class invokes LAMMPS to get the forces and energy in a given configuration.
    """
    def __init__(
        self,
        lammps_oracle_parameters: LammpsOracleParameters,
        element_types: ElementTypes,
        tmp_work_dir: Path,
        sw_coefficients_dir: Path = SW_COEFFICIENTS_DIR,
    ):
        """Init method.

        Args:
            lammps_oracle_parameters : parameters for the LAMMPS Oracle.
            element_types : object that knows how to transform element strings into ids and vice versa.
            tmp_work_dir : a temporary working directory.
            sw_coefficients_dir : the directory where the sw cofficient files can be found.
        """
        self._lammps_oracle_parameters = lammps_oracle_parameters
        self._element_types = element_types
        self.sw_coefficients_file_path = str(
            sw_coefficients_dir / self._lammps_oracle_parameters.sw_coeff_filename
        )
        self.tmp_work_dir = tmp_work_dir

        assert os.path.isfile(
            self.sw_coefficients_file_path
        ), f"The SW file '{self.sw_coefficients_file_path}' does not exist."

    def _create_lammps_commands(
        self,
        cartesian_positions: np.ndarray,
        box: np.ndarray,
        atom_types: np.ndarray,
        dump_file_path: Path,
    ) -> List[str]:
        commands = []
        commands.append("units metal")
        commands.append("atom_style atomic")
        commands.append(
            f"region simbox block 0 {box[0, 0]} 0 {box[1, 1]} 0 {box[2, 2]}"
        )
        commands.append(f"create_box {self._element_types.number_of_atom_types} simbox")
        commands.append("pair_style sw")

        elements_string = ""
        for element_id in self._element_types.element_ids:
            group_id = element_id + 1  # don't start the groups at zero
            element_name = self._element_types.get_element(element_id)
            elements_string += f" {element_name}"
            element_mass = Element(element_name).atomic_mass.real
            commands.append(f"group {element_name} type {group_id}")
            commands.append(f"mass {group_id} {element_mass}")

        commands.append(
            f"pair_coeff * * {self.sw_coefficients_file_path}{elements_string}"
        )

        for idx, cartesian_position in enumerate(cartesian_positions):
            element_id = atom_types[idx]
            group_id = element_id + 1  # don't start the groups at zero
            positions_string = " ".join(map(str, cartesian_position))
            commands.append(f"create_atoms {group_id} single {positions_string}")

        commands.append(
            "fix 1 all nvt temp 300 300 0.01"
        )  # selections here do not matter because we only do 1 step
        commands.append(f"dump 1 all yaml 1 {dump_file_path} id element x y z fx fy fz")
        commands.append(f"dump_modify 1 element {elements_string}")
        commands.append(
            "run 0"
        )  # 0 is the last step index - so run 0 means no MD update - just get the initial forces
        return commands

    def compute_energy_and_forces(
        self, cartesian_positions: np.ndarray, box: np.ndarray, atom_types: np.ndarray
    ):
        """Call LAMMPS to compute the energy and forces on all atoms in a configuration.

        Args:
            cartesian_positions: atomic positions in Euclidean space as a n_atom x spatial dimension array
            box: spatial dimension x spatial dimension array representing the periodic box. Assumed to be orthogonal.
            atom_types: n_atom array with an index representing the type of each atom
        Returns:
            energy: energy of configuration
            forces: forces on each atom in the configuration
        """
        assert np.allclose(
            box, np.diag(np.diag(box))
        ), "only orthogonal LAMMPS box are valid"

        dump_file_path = self.tmp_work_dir / "dump.yaml"

        # create a lammps run, turning off logging
        lmp = lammps.lammps(
            cmdargs=["-log", "none", "-echo", "none", "-screen", "none"]
        )
        commands = self._create_lammps_commands(
            cartesian_positions, box, atom_types, dump_file_path
        )
        for command in commands:
            lmp.command(command)

        # read information from lammps output
        with open(dump_file_path, "r") as f:
            dump_yaml = yaml.safe_load_all(f)
            doc = next(iter(dump_yaml))

        # clean up!
        dump_file_path.unlink()

        forces = pd.DataFrame(doc["data"], columns=doc["keywords"]).sort_values(
            "id"
        )  # organize in a dataframe

        # get the energy
        ke = lmp.get_thermo(
            "ke"
        )  # kinetic energy - should be 0 as atoms are created with 0 velocity
        pe = lmp.get_thermo("pe")  # potential energy
        energy = ke + pe

        return energy, forces

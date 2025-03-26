"""Call LAMMPS to get the forces and energy in a given configuration."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import lammps
import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Element

from diffusion_for_multi_scale_molecular_dynamics.oracle import \
    SW_COEFFICIENTS_DIR
from diffusion_for_multi_scale_molecular_dynamics.oracle.energy_oracle import (
    EnergyOracle, OracleParameters)


@dataclass(kw_only=True)
class LammpsOracleParameters(OracleParameters):
    """Lammps Oracle Parameters."""

    name: str = "lammps"
    sw_coeff_filename: str  # Stillinger-Weber potential filename


class LammpsEnergyOracle(EnergyOracle):
    """Lammps energy oracle.

    This class invokes LAMMPS to get the forces and energy in a given configuration.
    """

    def __init__(
        self,
        lammps_oracle_parameters: LammpsOracleParameters,
        sw_coefficients_dir: Path = SW_COEFFICIENTS_DIR,
    ):
        """Init method.

        Args:
            lammps_oracle_parameters : parameters for the LAMMPS Oracle.
            sw_coefficients_dir : the directory where the sw cofficient files can be found.
        """
        super().__init__(lammps_oracle_parameters)
        self.sw_coefficients_file_path = str(
            sw_coefficients_dir / lammps_oracle_parameters.sw_coeff_filename
        )

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

    def _compute_energy_and_forces(
        self,
        cartesian_positions: np.ndarray,
        box: np.ndarray,
        atom_types: np.ndarray,
        dump_file_path: Path,
    ):
        """Call LAMMPS to compute the energy and forces on all atoms in a configuration.

        Args:
            cartesian_positions: atomic positions in Euclidean space as a n_atom x spatial dimension array
            box: spatial dimension x spatial dimension array representing the periodic box. Assumed to be orthogonal.
            atom_types: n_atom array with an index representing the type of each atom
            dump_file_path: a temporary file where lammps will dump results.

        Returns:
            energy: energy of configuration
            forces: forces on each atom in the configuration
        """
        assert np.allclose(
            box, np.diag(np.diag(box))
        ), "only orthogonal LAMMPS box are valid"

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

    def _compute_one_configuration_energy_and_forces(
        self,
        cartesian_positions: np.ndarray,
        basis_vectors: np.ndarray,
        atom_types: np.ndarray,
    ) -> float:

        with tempfile.TemporaryDirectory() as tmp_work_dir:
            dump_file_path = Path(tmp_work_dir) / "dump.yaml"
            energy, forces = self._compute_energy_and_forces(
                cartesian_positions, basis_vectors, atom_types, dump_file_path
            )
            # clean up!
            dump_file_path.unlink()

        return energy, forces[["fx", "fy", "fz"]].to_numpy()

"""Call LAMMPS to get the forces and energy in a given configuration."""

import os
import warnings
from pathlib import Path
from typing import Dict, Tuple

import lammps
import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Element

from diffusion_for_multi_scale_molecular_dynamics.oracle import \
    SW_COEFFICIENTS_DIR


@warnings.deprecated("DO NOT USE THIS METHOD. It will be refactored away and replaced by LammpsEnergyOracle.")
def get_energy_and_forces_from_lammps(
    cartesian_positions: np.ndarray,
    box: np.ndarray,
    atom_types: np.ndarray,
    atom_type_map: Dict[int, str] = {1: "Si"},
    tmp_work_dir: str = "./",
    pair_coeff_dir: Path = SW_COEFFICIENTS_DIR,
) -> Tuple[float, pd.DataFrame]:
    """Call LAMMPS to compute the forces on all atoms in a configuration.

    Args:
        cartesian_positions: atomic positions in Euclidean space as a n_atom x spatial dimension array
        box: spatial dimension x spatial dimension array representing the periodic box. Assumed to be orthogonal.
        atom_types: n_atom array with an index representing the type of each atom
        atom_type_map (optional): map from index representing an atom type to a description of the atom.
            Defaults to {1: "Si"}
        tmp_work_dir (optional): directory where the LAMMPS output will be written (and deleted). Defaults to ./
        pair_coeff_dir (optional): directory where the potentials as .sw files are stored. Defaults to DATA_DIR

    Returns:
        energy
        dataframe with x, y, z coordinates and fx, fy, fz information in a dataframe.
    """
    n_atom = cartesian_positions.shape[0]
    assert atom_types.shape == (
        n_atom,
    ), f"Atom types should match the number of atoms. Got {atom_types.shape}."

    # create a lammps run, turning off logging
    lmp = lammps.lammps(cmdargs=["-log", "none", "-echo", "none", "-screen", "none"])
    assert np.allclose(
        box, np.diag(np.diag(box))
    ), "only orthogonal LAMMPS box are valid"

    lmp.command("units metal")
    lmp.command("atom_style atomic")

    lmp.command(
        f"region simbox block 0 {box[0, 0]} 0 {box[1, 1]} 0 {box[2, 2]}"
    )  # TODO what if box is not orthogonal
    lmp.command("create_box 1 simbox")
    lmp.command("pair_style sw")
    for k, v in atom_type_map.items():
        elem = Element(v)
        lmp.command(
            f"mass {k} {elem.atomic_mass.real}"
        )  # the .real is to get the value without the unit
        lmp.command(f"group {v} type {k}")
        lmp.command(
            f"pair_coeff * * {os.path.join(pair_coeff_dir, f'{v.lower()}.sw')} {v}"
        )
    for i in range(n_atom):
        lmp.command(
            f"create_atoms {atom_types[i]} single {' '.join(map(str, cartesian_positions[i, :]))}"
        )
    lmp.command(
        "fix 1 all nvt temp 300 300 0.01"
    )  # selections here do not matter because we only do 1 step
    # TODO not good in 2D
    lmp.command(
        f"dump 1 all yaml 1 {os.path.join(tmp_work_dir, 'dump.yaml')} id type x y z fx fy fz"
    )
    lmp.command(
        "run 0"
    )  # 0 is the last step index - so run 0 means no MD update - just get the initial forces

    # read information from lammps output
    with open(os.path.join(tmp_work_dir, "dump.yaml"), "r") as f:
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

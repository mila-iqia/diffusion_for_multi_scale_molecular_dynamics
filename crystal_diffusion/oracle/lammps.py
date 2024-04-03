import os
from typing import Dict

import lammps
import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Element

from crystal_diffusion import DATA_DIR


def get_forces_from_lammps(positions: np.ndarray,
                           box: np.ndarray,
                           atom_types: np.ndarray,
                           atom_type_map: Dict[int, str] = {1: 'Si'},
                           tmp_work_dir: str = './',
                           pair_coeff_dir: str = DATA_DIR) -> pd.DataFrame:
    """Call LAMMPS to compute the forces on all atoms in a configuration.

    Args:
        positions: atomic positions as a n_atom x spatial dimension array
        box: spatial dimension x spatial dimension array representing the periodic box. Assumed to be orthogonal.
        atom_types: n_atom array with an index representing the type of each atom
        atom_type_map (optional): map from index representing an atom type to a description of the atom.
            Defaults to {1: "Si"}
        tmp_work_dir (optional): directory where the LAMMPS output will be written (and deleted). Defaults to ./
        pair_coeff_dir (optional): directory where the potentials as .sw files are stored. Defaults to DATA_DIR

    Returns:
        dataframe with x, y, z coordinates and fx, fy, fz information in a dataframe.
    """
    n_atom = positions.shape[0]
    assert atom_types.shape == (n_atom, ), f"Atom types should match the number of atoms. Got {atom_types.shape}."
    lmp = lammps.lammps()  # create a lammps run
    lmp.command(f"region simbox block 0 {box[0, 0]} 0 {box[1, 1]} 0 {box[2, 2]}")  # TODO what if box is not orthogonal
    lmp.command("create_box 1 simbox")
    lmp.command("pair_style sw")
    for k, v in atom_type_map.items():
        elem = Element(v)
        lmp.command(f"mass {k} {elem.atomic_mass.real}")  # the .real is to get the value without the unit
        lmp.command(f"group {v} type {k}")
        lmp.command(f"pair_coeff * * {os.path.join(pair_coeff_dir, f'{v.lower()}.sw')} {v}")
    for i in range(n_atom):
        lmp.command(f"create_atoms {atom_types[i]} single {' '.join(map(str, positions[i, :]))}")
    lmp.command("fix 1 all nvt temp 300 300 0.01")  # selections here do not matter because we only do 1 step
    lmp.command(f"dump 1 all yaml 1 {os.path.join(tmp_work_dir, 'dump.yaml')} id x y z fx fy fz")  # TODO not good in 2D
    lmp.command("run 0")  # 0 is the last step index - so run 0 means no MD update - just get the initial forces

    # read informations from lammps output
    with open(os.path.join(tmp_work_dir, "dump.yaml"), "r") as f:
        dump_yaml = yaml.safe_load_all(f)
        doc = next(iter(dump_yaml))
    os.remove(os.path.join(tmp_work_dir, "dump.yaml"))  # no need to keep the output as arteface
    forces = pd.DataFrame(doc['data'], columns=doc['keywords']).sort_values("id")  # organize in a dataframe
    return forces

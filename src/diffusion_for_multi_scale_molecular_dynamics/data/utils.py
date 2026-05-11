"""Utility functions for data processing."""

import logging
import os
from typing import Any, AnyStr, Dict, List, Tuple

import numpy as np
import yaml
from ase import Atoms
from yaml import CDumper, CLoader

from ..namespace import AXL

logger = logging.getLogger(__name__)


def crop_lammps_yaml(
    lammps_dump: str, lammps_thermo: str, crop_step: int, inplace: bool = False
) -> Tuple[List[Dict[AnyStr, Any]], Dict[AnyStr, Any]]:
    """Remove the first steps of a LAMMPS run to remove structures near the starting point.

    Args:
        lammps_dump: path to LAMMPS output file as a yaml
        lammps_thermo: path to LAMMPS thermodynamic output file as a yaml
        crop_step: number of steps to remove
        inplace (optional): if True, overwrite the two LAMMPS file with a cropped version. If False, do not write.
            Defaults to False.

    Returns:
        cropped LAMMPS output file
        cropped LAMMPS thermodynamic output file
    """
    if not os.path.exists(lammps_dump):
        raise ValueError(
            f"{lammps_dump} does not exist. Please provide a valid LAMMPS dump file as yaml."
        )

    if not os.path.exists(lammps_thermo):
        raise ValueError(
            f"{lammps_thermo} does not exist. Please provide a valid LAMMPS thermo log file as yaml."
        )

    # get the atom information (positions and forces) from the LAMMPS 'dump' file
    with open(lammps_dump, "r") as f:
        logger.info("loading dump file....")
        dump_yaml = yaml.load_all(f, Loader=CLoader)
        logger.info("creating list of documents...")
        dump_yaml = [d for d in dump_yaml]  # generator to list
    # every MD iteration is saved as a separate document in the yaml file
    # prepare a dataframe to get all the data
    if crop_step >= len(dump_yaml):
        raise ValueError(
            f"Trying to remove {crop_step} steps in a run of {len(dump_yaml)} steps."
        )
    logger.info("cropping documents...")
    dump_yaml = dump_yaml[crop_step:]

    # get the total energy from the LAMMPS thermodynamic output
    with open(lammps_thermo, "r") as f:
        logger.info("loading thermo file....")
        thermo_yaml = yaml.load(f, Loader=CLoader)
    logger.info("cropping thermo file....")
    thermo_yaml["data"] = thermo_yaml["data"][crop_step:]

    if inplace:
        with open("test_yaml.yaml", "w") as f:
            yaml.dump_all(dump_yaml, f, explicit_start=True, Dumper=CDumper)
        with open("test_thermo.yaml", "w") as f:
            yaml.dump(thermo_yaml, f, Dumper=CDumper)

    return dump_yaml, thermo_yaml


def traj_to_AXL(traj, element_list):
    """Convert a traj or a list of atoms to the AXL format.

    Args:
        traj: a list of Atoms objects or a Trajectory object
        element_list : list[str] containing the symbols of each element
    Returns:
        A list of AXL namedtuples with A, X, L being np.array
         A: Atomic types index in element_list
         X: Relative coordinates of the atoms
         L: Lattice parameters in Voigt Notation
    """
    axl_list = []
    for atoms in traj:
        A = np.array([element_list.index(sym) for sym in atoms.get_chemical_symbols()], dtype=np.int64)
        X = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=np.float32)
        c = np.asarray(atoms.cell.array, dtype=np.float32)  # Lattice must follow Voigt Notation
        L = np.array([c[0, 0], c[1, 1], c[2, 2], c[1, 2], c[0, 2], c[0, 1]], dtype=np.float32)
        axl_list.append(AXL(A=A, X=X, L=L))
    return axl_list


def AXL_to_traj(axl_list, element_list):
    """Convert a list of AXL format to a list of Atoms.

    Args:
        axl_list : A list of AXL namedtuples with A, X, L
          A: Atomic types index in element_list
          X: Relative coordinates of the atoms
          L: Lattice parameters in Voigt Notation
        element_list : list[str] containing the symbols of each element
    Returns:
        A list of Atoms objects
    """
    atoms_list = []
    for axl in axl_list:
        A = axl.A
        X = axl.X
        L = axl.L
        c00, c11, c22, c12, c02, c01 = L
        cell = np.array(
            [[c00, c01, c02],
             [c01, c11, c12],
             [c02, c12, c22]],
            dtype=np.float32,
        )
        symbols = [element_list[int(t)] for t in A]
        atoms = Atoms(
            symbols=symbols,
            cell=cell,
            pbc=True,
            scaled_positions=X,
        )
        atoms_list.append(atoms)
    return atoms_list

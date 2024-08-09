import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from pymatgen.core import Structure


@dataclass(kw_only=True)
class MTPInputs:
    """Create a dataclass to train or evaluate a MTP model."""
    structure: List[Structure]
    forces: List[List[float]]
    energy: List[float]


def extract_structure_and_forces_from_file(filename: str, atom_dict: Dict[int, Any]) -> \
        Tuple[List[Structure], List[List[float]]]:
    """Convert LAMMPS yaml output in a format compatible with MTP training and evaluation methods.

    Args:
        filename: path to LAMMPS output file in yaml format
        atom_dict: mapping from LAMMPS atom indices to atom type (atomic number as int or atom name as str)

    Returns:
        list of pymatgen Structure containing the atoms and their positions
        list of forces (n x 3) for each atom
    """
    structures = []
    forces = []
    with (open(filename, 'r') as f):
        l_yaml = yaml.safe_load_all(f)
        for d in l_yaml:  # loop over LAMMPS outputs and convert in pymatgen Structure objects
            # lattice in yaml is 3 x 2 [0, x_lim]
            # we assume a rectangular lattice for now with the 2nd coordinates as the lattice vectors
            lattice = np.zeros((3, 3))
            for i, x in enumerate(d['box']):
                lattice[i, i] = x[1]
            type_idx = d['keywords'].index('type')
            species = [atom_dict[x[type_idx]] for x in d['data']]  # convert to atom type
            coords_idx = [d['keywords'].index(x) for x in ['x', 'y', 'z']]
            coords = [[x[i] for i in coords_idx] for x in d['data']]
            pm_structure = Structure(lattice=lattice,
                                     species=species,
                                     coords=coords,
                                     coords_are_cartesian=True)
            structures.append(pm_structure)
            force_idx = [d['keywords'].index(x) for x in ['fx', 'fy', 'fz']]
            structure_forces = [[x[i] for i in force_idx] for x in d['data']]
            forces.append(structure_forces)
    return structures, forces


def extract_energy_from_thermo_log(filename: str) -> List[float]:
    """Read energies from LAMMPS thermodynamic output file.

    Args:
        filename: path to LAMMPS thermodynamic output file in yaml format.

    Returns:
        list of energies (1 value per configuration)
    """
    with open(filename, 'r') as f:
        log_yaml = yaml.safe_load(f)
        kin_idx = log_yaml['keywords'].index('KinEng')
        pot_idx = log_yaml['keywords'].index('PotEng')
        energies = [x[kin_idx] + x[pot_idx] for x in log_yaml['data']]
    return energies


def prepare_mtp_inputs_from_lammps(output_yaml: List[str],
                                   thermo_yaml: List[str],
                                   atom_dict: Dict[int, Any]
                                   ) -> MTPInputs:
    """Convert a list of LAMMPS output files and thermodynamic output files to MTP input format.

    Args:
        output_yaml: list of LAMMPS output files as yaml.
        thermo_yaml: list of LAMMPS thermodynamic output files as yaml.
        atom_dict: mapping of LAMMPS indices to atom type.

    Returns:
        dataclass used to
    """
    mtp_inputs = {
        'structure': [],
        'energy': [],
        'forces': []
    }
    for filename in output_yaml:
        structures, forces = extract_structure_and_forces_from_file(filename, atom_dict)
        mtp_inputs['structure'] += structures
        mtp_inputs['forces'] += forces
    for filename in thermo_yaml:
        mtp_inputs['energy'] += extract_energy_from_thermo_log(filename)
    mtp_inputs = MTPInputs(structure=mtp_inputs['structure'],
                           energy=mtp_inputs['energy'],
                           forces=mtp_inputs['forces'])
    return mtp_inputs


def crawl_lammps_directory(folder_name: str, folder_name_pattern: str= "train") -> Tuple[List[str], List[str]]:
    """Crawl through a folder and find the LAMMPS output files in folders containing a specified pattern in their name.

    LAMMPS outputs should end with dump.yaml and Thermondynamics variables files should end with thermo.yaml

    Args:
        folder_name: folder to crawl
        folder_name_pattern (optional): name of the subfolder to keep. Defaults to train.

    Returns:
        list of LAMMPS dump outputs and list of LAMMPS thermo outputs

    """
    assert os.path.exists(folder_name), "Invalid folder name provided."
    lammps_output_files, thermo_output_files = [], []
    for dirpath, _, filenames in os.walk(folder_name):
        if re.search(folder_name_pattern, dirpath):
            lammps_output_files.extend([os.path.join(dirpath, f) for f in filenames if f.endswith("dump.yaml")])
            thermo_output_files.extend([os.path.join(dirpath, f) for f in filenames if f.endswith("thermo.yaml")])
    return lammps_output_files, thermo_output_files

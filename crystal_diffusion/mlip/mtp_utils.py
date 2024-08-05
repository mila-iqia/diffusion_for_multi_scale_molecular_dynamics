from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from pymatgen.core import Structure


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

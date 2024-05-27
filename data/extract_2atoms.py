"""Script to extract 2 atoms from a 1x1x1 cubic cell (8 atoms)."""
import os
import shutil
from typing import Any, AnyStr, Dict, List

import numpy as np
import yaml
from yaml import CDumper, CLoader


# general variable for Si lattice
lattice_parameter = 5.43  # lattice parameter of Si in Angstrom
red_position = np.array([[0, 0, 0], [0.25, 0.25, 0.25 ]])  # https://www.physics-in-a-nutshell.com/article/13/diamond-structure
fcc_vectors = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
si_atom_pos = [np.matmul(lattice_parameter * fcc_vectors, x) for x in red_position]

src_path = 'si_diffusion_1x1x1/'
n_train = 10
n_valid = 5
dst_path = 'si_diffusion_2atoms'


def read_yaml(file_path: str) -> List[Dict[AnyStr, Any]]:
    """Read LAMMPS dump file as yaml format.

    Args:
        file_path: path to file to read

    Returns:
        file information as a list. Each entry is a MD timestep.
    """
    with open(file_path, 'r') as f:
        dump_yaml = yaml.load_all(f, Loader=CLoader)
        dump_yaml = [d for d in dump_yaml]  # generator to list
    return dump_yaml


def get_distance_squared(atom_pos: np.array, target_pos: np.array) -> float:
    """Find the distance between an atom and a target position, considering shifts by a lattice vector.

    Args:
        atom_pos: atom position as spatial_dim array
        target_pos: target position as spatial_dim array

    Returns:
        minimal distance to target considering shifts by a lattice vector
    """
    distance = [np.sum((atom_pos - target_pos) ** 2),
                np.sum((atom_pos + lattice_parameter * fcc_vectors[0] - target_pos) ** 2),
                np.sum((atom_pos - lattice_parameter * fcc_vectors[0] - target_pos) ** 2),
                np.sum((atom_pos + lattice_parameter * fcc_vectors[1] - target_pos) ** 2),
                np.sum((atom_pos - lattice_parameter * fcc_vectors[1] - target_pos) ** 2),
                np.sum((atom_pos + lattice_parameter * fcc_vectors[2] - target_pos) ** 2),
                np.sum((atom_pos - lattice_parameter * fcc_vectors[2] - target_pos) ** 2),]
    return min(distance)


def find_atom_closest_to_target(data: List[List[float]], target_pos: np.array) -> int:
    """Get index of atom closest to target position.

    Args:
        data: list of atomic positions. Should be N atom x spatial_dimension
        target_pos: target position. Should be spatial_dimension

    Returns:
        index of the closest atom to the target
    """
    positions = np.array(data)[:, 2:5]
    # get distance considering lattice vector shift
    # distance_squared = [get_distance_squared(atom_pos, target_pos) for atom_pos in positions]
    # or ignore them and get closest to (0, 0, 0), (1/4, 1/4, 1/4)
    distance_squared = np.sum((positions - target_pos) ** 2, axis=1)
    return np.argmin(distance_squared)


def crop_atoms_from_yaml(yaml_dump: Dict[AnyStr, Any]) -> Dict[AnyStr, Any]:
    """Rewrite yaml to have 2 atoms only.

    Args:
        yaml_dump: parsed yaml file with 8 atoms

    Returns:
        parsed yaml file with 2 atoms
    """
    atoms = yaml_dump["data"]
    atom_keep = [find_atom_closest_to_target(atoms, x) for x in si_atom_pos]
    # keep the two atoms closest to 0,0,0 and 1/4, 1/4, 1/4
    yaml_dump["data"] = [atoms[atom] for atom in atom_keep]
    # reindex them to be atoms 1 and 2 (original yamls do not start indexing at 0)
    yaml_dump["data"] = [[i + 1] + d[1:] for i, d in enumerate(yaml_dump["data"])]
    yaml_dump["natoms"] = len(atom_keep)
    # replace box variable with flatten lattice vectors instead
    del yaml_dump["box"]
    yaml_dump["lattice"] = (lattice_parameter * fcc_vectors).flatten().tolist()
    return yaml_dump


def parse_yaml(yaml_path: str) -> List[Dict[AnyStr, Any]]:
    """Read a LAMMPS dump and isolate 2 atoms in each time step.

    Args:
        yaml_path: path to a yaml file

    Returns:
        list of MD timesteps with 2 atoms only
    """
    yaml_dump = read_yaml(yaml_path)
    new_yaml = []
    for d in yaml_dump:
        new_yaml.append(crop_atoms_from_yaml(d))
    return new_yaml


def main():
    for i in range(1, n_train + n_valid + 1):
        mode = 'train' if i < n_train + 1 else 'valid'
        src_yaml = os.path.join(src_path, f'{mode}_run_{i}')
        dst_yaml = os.path.join(dst_path, f'{mode}_run_{i}')
        if not os.path.exists(dst_yaml):
            os.makedirs(dst_yaml)
        crop_yaml = parse_yaml(os.path.join(src_yaml, 'lammps_dump.yaml'))
        with open(os.path.join(dst_yaml, 'lammps_dump.yaml'), "w") as f:
            yaml.dump_all(crop_yaml, f, explicit_start=True, Dumper=CDumper)
        shutil.copyfile(os.path.join(src_yaml, 'lammps_thermo.yaml'), os.path.join(dst_yaml, 'lammps_thermo.yaml'))


if __name__ == '__main__':
    main()

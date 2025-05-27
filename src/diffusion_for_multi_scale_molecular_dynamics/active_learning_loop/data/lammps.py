"""LAMMPS.

This module implements methods to read LAMMPS dump output. It is assumed that the dump files are in the
yaml format, and that the thermo data is included in the same file.

The aim of this module is to extract single point calculations for further processing. This is somewhat
different from what is done in src/diffusion_for_multi_scale_molecular_dynamics/data/parse_lammps_outputs.py,
where there the goal is to combine the data into a parquet archive for training a generative model.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Lattice, Structure
from tqdm import tqdm
from yaml import CLoader

_ID_FIELD = "id"  # the atom id
_ELEMENT_FIELD = "element"
_POSITIONS_FIELDS = ["x", "y", "z"]  # the atomic cartesian positions
_FORCES_FIELDS = ["fx", "fy", "fz"]  # the atomic forces
_BOX_FIELD = "box"
_ENERGY_FIELD = "PotEng"


def _extract_data_from_yaml_document(yaml_document: dict) -> Tuple[pd.DataFrame, Dict]:
    """Extract data from yaml document.

    We make the assumption that the content of the dump file is for 3D data.

    Args:
        yaml_document: A single document, assumed to contain both global ("thermo") information and
            atomwise information.

    Returns:
        atoms_df: a dataframe with all the atomwise information.
        global_dict: a dictionary with the global information.
    """
    columns = yaml_document['keywords']
    data = yaml_document['data']
    atoms_df = pd.DataFrame(data=data, columns=columns).sort_values(by=_ID_FIELD)

    global_dict = _parse_thermo_fields(yaml_document)
    # We assume an orthogonal cell
    global_dict['cell_dimensions'] = np.array([bounds[1] for bounds in yaml_document[_BOX_FIELD]])

    return atoms_df, global_dict


def _parse_thermo_fields(yaml_document: Dict) -> Dict:
    """Parse thermo fields.

    Args:
        yaml_document: A single document, assumed to contain both global ("thermo") information and
            atomwise information.

    Returns:
        thermo_dict: a dictionary with the thermo fields.
    """
    assert 'thermo' in yaml_document, "The input document does not have the keyword thermo"
    keywords = yaml_document['thermo'][0]['keywords']
    data = yaml_document['thermo'][1]['data']
    results = {k: v for k, v in zip(keywords, data)}
    return results


def _get_structure_from_atoms_dataframe(atoms_df: pd.DataFrame, cell_dimensions: np.ndarray) -> Structure:
    """Get structure from atoms dataframe.

    Extract a pymatgen structure from the atoms dataframe.

    Args:
        atoms_df: a dataframe with all the atomwise information.
        cell_dimensions: a numpy array containing the dimensions of the orthorhombic cell.

    Returns:
        structure: the corresponding pymatgen structure
    """
    lattice = Lattice(matrix=np.diag(cell_dimensions), pbc=(True, True, True))

    structure = Structure(lattice=lattice,
                          species=atoms_df[_ELEMENT_FIELD].values,
                          coords=atoms_df[_POSITIONS_FIELDS].values,
                          coords_are_cartesian=True,
                          )

    return structure


def _get_forces_from_atoms_dataframe(atoms_df: pd.DataFrame) -> np.ndarray:
    """Get forces from atoms dataframe."""
    return atoms_df[_FORCES_FIELDS].values


def extract_structures_forces_and_energies_from_dump(lammps_dump_path: Path) -> (
        Tuple)[List[Structure], List[np.ndarray], List[float]]:
    """Extract structures, forces and energies from lammps dump.

    Args:
        lammps_dump_path: path to a lammps dump file, in yaml format, assumed to also contain the thermo data.

    Returns:
        list_structures: the structures in the dump file.
        list_forces: the forces in the dump file, in the same order as the structures.
        list_energies: the energies in the dump file, in the same order as the structures.
    """
    list_structures = []
    list_forces = []
    list_energies = []
    with open(str(lammps_dump_path), "r") as stream:
        for yaml_document in tqdm(yaml.load_all(stream, Loader=CLoader), "PARSING YAML"):
            atoms_df, global_data_dict = _extract_data_from_yaml_document(yaml_document)
            cell_dimensions = global_data_dict['cell_dimensions']
            structure = _get_structure_from_atoms_dataframe(atoms_df, cell_dimensions)
            list_structures.append(structure)

            forces = _get_forces_from_atoms_dataframe(atoms_df)
            list_forces.append(forces)

            list_energies.append(global_data_dict[_ENERGY_FIELD])

    return list_structures, list_forces, list_energies

"""LAMMPS.

This module implements methods to read LAMMPS dump output. It is assumed that the dump files are in the
yaml format, and that the thermo data is included in the same file.

The aim of this module is to extract single point calculations for further processing. This is somewhat
different from what is done in src/diffusion_for_multi_scale_molecular_dynamics/data/parse_lammps_outputs.py,
where there the goal is to combine the data into a parquet archive for training a generative model.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Lattice, Structure
from tqdm import tqdm
from yaml import CLoader

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.namespace import (
    BOX_FIELD, ELEMENT_FIELD, ENERGY_FIELD, FORCES_FIELDS, ID_FIELD,
    POSITIONS_FIELDS, UNCERTAINTY_FIELD)


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
    columns = yaml_document["keywords"]
    data = yaml_document["data"]
    atoms_df = pd.DataFrame(data=data, columns=columns).sort_values(by=ID_FIELD)

    global_dict = _parse_thermo_fields(yaml_document)
    # We assume an orthogonal cell
    global_dict["cell_dimensions"] = np.array(
        [bounds[1] for bounds in yaml_document[BOX_FIELD]]
    )

    return atoms_df, global_dict


def _parse_thermo_fields(yaml_document: Dict) -> Dict:
    """Parse thermo fields.

    Args:
        yaml_document: A single document, assumed to contain both global ("thermo") information and
            atomwise information.

    Returns:
        thermo_dict: a dictionary with the thermo fields.
    """
    assert (
        "thermo" in yaml_document
    ), "The input document does not have the keyword thermo"
    keywords = yaml_document["thermo"][0]["keywords"]
    data = yaml_document["thermo"][1]["data"]
    results = {k: v for k, v in zip(keywords, data)}
    return results


def _get_structure_from_atoms_dataframe(
    atoms_df: pd.DataFrame, cell_dimensions: np.ndarray
) -> Structure:
    """Get structure from atoms dataframe.

    Extract a pymatgen structure from the atoms dataframe.

    Args:
        atoms_df: a dataframe with all the atomwise information.
        cell_dimensions: a numpy array containing the dimensions of the orthorhombic cell.

    Returns:
        structure: the corresponding pymatgen structure
    """
    lattice = Lattice(matrix=np.diag(cell_dimensions), pbc=(True, True, True))

    structure = Structure(
        lattice=lattice,
        species=atoms_df[ELEMENT_FIELD].values,
        coords=atoms_df[POSITIONS_FIELDS].astype(float).values,
        coords_are_cartesian=True,
    )

    return structure


def _get_forces_from_atoms_dataframe(atoms_df: pd.DataFrame) -> np.ndarray:
    """Get forces from atoms dataframe."""
    return atoms_df[FORCES_FIELDS].astype(float).values


def _get_uncertainties_from_atoms_dataframe(
    atoms_df: pd.DataFrame,
) -> Union[np.ndarray, None]:
    """Get uncertainties from atoms dataframe."""
    if UNCERTAINTY_FIELD in atoms_df.columns:
        return atoms_df[UNCERTAINTY_FIELD].astype(float).values
    else:
        return None


def extract_all_fields_from_dump(
    lammps_dump_path: Path,
) -> (Tuple)[
    List[Structure], List[np.ndarray], List[float], List[Union[np.ndarray, None]]
]:
    """Extract structures, forces and energies from lammps dump.

    Args:
        lammps_dump_path: path to a lammps dump file, in yaml format, assumed to also contain the thermo data.

    Returns:
        list_structures: the structures in the dump file.
        list_forces: the forces in the dump file, in the same order as the structures.
        list_energies: the energies in the dump file, in the same order as the structures.
        list_uncertainties: the uncertainties in the dump file, if present, else None.
    """
    list_structures = []
    list_forces = []
    list_energies = []
    list_uncertainties = []
    with open(str(lammps_dump_path), "r") as stream:
        for yaml_document in tqdm(
            yaml.load_all(stream, Loader=CLoader), "PARSING YAML"
        ):
            atoms_df, global_data_dict = _extract_data_from_yaml_document(yaml_document)
            cell_dimensions = global_data_dict["cell_dimensions"]
            structure = _get_structure_from_atoms_dataframe(atoms_df, cell_dimensions)
            list_structures.append(structure)

            forces = _get_forces_from_atoms_dataframe(atoms_df)
            list_forces.append(forces)

            list_energies.append(global_data_dict[ENERGY_FIELD])
            list_uncertainties.append(_get_uncertainties_from_atoms_dataframe(atoms_df))

    return list_structures, list_forces, list_energies, list_uncertainties

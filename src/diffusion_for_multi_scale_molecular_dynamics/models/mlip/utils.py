import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Structure
from sklearn.metrics import mean_absolute_error


@dataclass(kw_only=True)
class MLIPInputs:
    """Create a dataclass to train or evaluate a MTP model."""

    structure: List[Structure]
    forces: List[List[List[float]]]  # num samples x num atoms x spatial dimension
    energy: List[float]


def extract_structure_and_forces_from_file(
    filename: str, atom_dict: Dict[int, Any], forces_avail: bool = True
) -> Tuple[List[Structure], Optional[List[List[float]]]]:
    """Convert LAMMPS yaml output in a format compatible with MTP training and evaluation methods.

    Args:
        filename: path to LAMMPS output file in yaml format
        atom_dict: mapping from LAMMPS atom indices to atom type (atomic number as int or atom name as str)
        forces_avail (optional): if True, get the forces from the LAMMPS output file. Defaults to True.

    Returns:
        list of pymatgen Structure containing the atoms and their positions
        list of forces (n x 3) for each atom. None if forces_avail is False
    """
    structures = []
    forces = [] if forces_avail else None
    with open(filename, "r") as f:
        l_yaml = yaml.safe_load_all(f)
        for (
            d
        ) in (
            l_yaml
        ):  # loop over LAMMPS outputs and convert in pymatgen Structure objects
            # lattice in yaml is 3 x 2 [0, x_lim]
            # we assume a rectangular lattice for now with the 2nd coordinates as the lattice vectors
            lattice = np.zeros((3, 3))
            for i, x in enumerate(d["box"]):
                lattice[i, i] = x[1]
            type_idx = d["keywords"].index("element")
            species = [x[type_idx] for x in d["data"]]
            coords_idx = [d["keywords"].index(x) for x in ["x", "y", "z"]]
            coords = [[x[i] for i in coords_idx] for x in d["data"]]
            pm_structure = Structure(
                lattice=lattice,
                species=species,
                coords=coords,
                coords_are_cartesian=True,
            )
            structures.append(pm_structure)
            if forces_avail:
                force_idx = [d["keywords"].index(x) for x in ["fx", "fy", "fz"]]
                structure_forces = [[x[i] for i in force_idx] for x in d["data"]]
                forces.append(structure_forces)
    return structures, forces


def extract_energy_from_thermo_log(filename: str) -> List[float]:
    """Read energies from LAMMPS thermodynamic output file.

    Args:
        filename: path to LAMMPS thermodynamic output file in yaml format.

    Returns:
        list of energies (1 value per configuration)
    """
    with open(filename, "r") as f:
        log_yaml = yaml.safe_load(f)
        kin_idx = log_yaml["keywords"].index("KinEng")
        pot_idx = log_yaml["keywords"].index("PotEng")
        energies = [x[kin_idx] + x[pot_idx] for x in log_yaml["data"]]
    return energies


def prepare_mlip_inputs_from_lammps(
    output_yaml: List[str],
    thermo_yaml: List[str],
    atom_dict: Dict[int, Any],
    get_forces: bool = True,
) -> MLIPInputs:
    """Convert a list of LAMMPS output files and thermodynamic output files to MTP input format.

    Args:
        output_yaml: list of LAMMPS output files as yaml.
        thermo_yaml: list of LAMMPS thermodynamic output files as yaml.
        atom_dict: mapping of LAMMPS indices to atom type.
        get_forces (optional): if True, get the forces. Defaults to True.

    Returns:
        dataclass used as inputs to train and evaluation a MTP model
    """
    mtp_inputs = {"structure": [], "energy": [], "forces": []}
    for filename in output_yaml:
        structures, forces = extract_structure_and_forces_from_file(
            filename, atom_dict, get_forces
        )
        mtp_inputs["structure"] += structures
        mtp_inputs["forces"] += forces  # will be None if get_forces is False
    for filename in thermo_yaml:
        mtp_inputs["energy"] += extract_energy_from_thermo_log(filename)
    mtp_inputs = MLIPInputs(
        structure=mtp_inputs["structure"],
        energy=mtp_inputs["energy"],
        forces=mtp_inputs["forces"],
    )
    return mtp_inputs


def crawl_lammps_directory(
    folder_name: str, folder_name_pattern: str = "train"
) -> Tuple[List[str], List[str]]:
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
            lammps_output_files.extend(
                [os.path.join(dirpath, f) for f in filenames if f.endswith("dump.yaml")]
            )
            thermo_output_files.extend(
                [
                    os.path.join(dirpath, f)
                    for f in filenames
                    if f.endswith("thermo.yaml")
                ]
            )
    return lammps_output_files, thermo_output_files


def concat_mlip_inputs(input1: MLIPInputs, input2: MLIPInputs) -> MLIPInputs:
    """Merge two MLIP inputs data class.

    Args:
        input1: first MLIPInputs dataset
        input2: second MLIPInputs dataset

    Returns:
        concatenated MTPInputs dataset
    """
    concat_inputs = MLIPInputs(
        structure=input1.structure + input2.structure,
        forces=input1.forces + input2.forces,
        energy=input1.energy + input2.energy,
    )
    return concat_inputs


def get_metrics_from_pred(
    df_orig: pd.DataFrame, df_predict: pd.DataFrame
) -> Tuple[float, float]:
    """Get mean absolute error on energy and forces from the outputs of MTP.

    Args:
        df_orig: dataframe with ground truth values
        df_predict: dataframe with MTP predictions

    Returns:
        MAE on energy in eV/atom and MAE on forces in eV/Å
    """
    # from demo in maml
    # get a single predicted energy per structure
    predicted_energy = df_predict.groupby("structure_index").agg(
        {"energy": "mean", "atom_index": "count"}
    )
    # normalize by number of atoms
    predicted_energy = (
        predicted_energy["energy"] / predicted_energy["atom_index"]
    ).to_numpy()
    # same for ground truth
    gt_energy = df_orig.groupby("structure_index").agg(
        {"energy": "mean", "atom_index": "count"}
    )
    gt_energy = (gt_energy["energy"] / gt_energy["atom_index"]).to_numpy()

    predicted_forces = df_predict[["fx", "fy", "fz"]].to_numpy().flatten()
    gt_forces = df_orig[["fx", "fy", "fz"]].to_numpy().flatten()

    return mean_absolute_error(predicted_energy, gt_energy), mean_absolute_error(
        predicted_forces, gt_forces
    )

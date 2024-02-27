from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from maml.apps.pes import MTPotential
from monty.serialization import loadfn
from pymatgen.core import Structure
from sklearn.metrics import mean_absolute_error


# TODO list of yaml files should come from an external call
# yaml dump file
lammps_yaml = ['lammps_scripts/Si/si-custom/dump.si-300-1.yaml']
# yaml thermodynamic variables
lammps_thermo_yaml = ['lammps_scripts/Si/si-custom/thermo_log.yaml']
# note that the YAML output does not contain the map from index to atomic species
# this will have to be taken from elsewhere
# use a manual map for now
atom_dict = {1: 'Si'}


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
            PMStructure = Structure(lattice=lattice,
                                    species=species,
                                    coords=coords)
            structures.append(PMStructure)
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


def prepare_mtp_inputs_from_lammps(output_yaml: List[str], thermo_yaml: List[str], atom_dict: Dict[int, Any]) -> \
    Dict[str, Any]:
    """Convert a list of LAMMPS output files and thermodynamic output files to MTP input format.

    Args:
        output_yaml: list of LAMMPS output files as yaml.
        thermo_yaml: list of LAMMPS thermodynamic output files as yaml.
        atom_dict: mapping of LAMMPS indices to atom type.

    Returns:
        dict with structure, energies and forces usable by MTP.
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
    return mtp_inputs


def train_mtp(train_inputs: Dict[str, Any], valid_inputs: Dict[str, Any], max_dist: float=5) -> \
    Tuple[pd.DataFrame, pd.DataFrame]:
    """Create and evaluate a MTP potential.

    Args:
        train_inputs: inputs for training. Should contain structure, energies and forces
        valid_inputs: inputs for validation.
        max_dist (optional): radial cutoff. Defaults to 5.

    Returns:
       dataframe with original and predicted energies and forces.
    """
    # TODO more kwargs for MTP training. See maml documentation.
    # create MTP
    mtp = MTPotential()

    # train
    mtp.train(
        train_structures=train_inputs["structure"],
        train_energies=train_inputs["energy"],
        train_forces=train_inputs["forces"],
        train_stresses=None,
        max_dist=5,
        stress_weight=0,
    )

    # evaluate
    df_orig, df_predict = mtp.evaluate(
        test_structures=train_inputs["structure"],
        test_energies=train_inputs["energy"],
        test_forces=train_inputs["forces"],
        test_stresses=None,
    )

    # TODO also output the MTP
    return df_orig, df_predict


def main():
    mtp_inputs = prepare_mtp_inputs_from_lammps(lammps_yaml, lammps_thermo_yaml, atom_dict)
    df_orig, df_predict = train_mtp(mtp_inputs, mtp_inputs)
    # from demo in maml
    E_p = np.array(df_predict[df_predict["dtype"] == "energy"]["y_orig"])
    E_p /= df_predict[df_predict["dtype"] == "energy"]["n"]
    E_o = np.array(df_orig[df_orig["dtype"] == "energy"]["y_orig"])
    E_o /= df_orig[df_orig["dtype"] == "energy"]["n"]
    print(f"MAE of training energy prediction is {mean_absolute_error(E_o, E_p) * 1000} meV/atom")

    F_p = np.array(df_predict[df_predict["dtype"] == "force"]["y_orig"])
    F_p /= df_predict[df_predict["dtype"] == "force"]["n"]
    F_o = np.array(df_orig[df_orig["dtype"] == "force"]["y_orig"])
    F_o /= df_orig[df_orig["dtype"] == "force"]["n"]
    print(f"MAE of training force prediction is {mean_absolute_error(F_o, F_p)} eV/Å")


if __name__ == '__main__':
    main()

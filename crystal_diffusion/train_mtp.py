"""Script to train and evaluate a MTP.

Running the main() runs a debugging example. Entry points are train_mtp and evaluate_mtp.
"""
import argparse
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Structure
from sklearn.metrics import mean_absolute_error

from crystal_diffusion.models.mtp import MTPWithMLIP3

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


class MTP_Inputs(NamedTuple):
    """Create a namedtuple instance for MTP inputs."""

    structure: List[Structure]
    forces: List[List[float]]
    energy: List[float]


def prepare_mtp_inputs_from_lammps(output_yaml: List[str],
                                   thermo_yaml: List[str],
                                   atom_dict: Dict[int, Any]
                                   ) -> MTP_Inputs:
    """Convert a list of LAMMPS output files and thermodynamic output files to MTP input format.

    Args:
        output_yaml: list of LAMMPS output files as yaml.
        thermo_yaml: list of LAMMPS thermodynamic output files as yaml.
        atom_dict: mapping of LAMMPS indices to atom type.

    Returns:
        namedtuple with structure, energies and forces usable by MTP.
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
    mtp_inputs = MTP_Inputs(structure=mtp_inputs['structure'],
                            energy=mtp_inputs['energy'],
                            forces=mtp_inputs['forces'])
    return mtp_inputs


def train_mtp(train_inputs: MTP_Inputs, mlip_folder_path: str, save_dir: str) -> MTPWithMLIP3:
    """Create and train an MTP potential.

    Args:
        train_inputs: inputs for training. Should contain structure, energies and forces
        mlip_folder_path: path to MLIP-3 folder
        save_dir: path to directory where to save the fitted model

    Returns:
       dataframe with original and predicted energies and forces.
    """
    # TODO more kwargs for MTP training. See maml and mlip-3 documentation.
    # create MTP
    mtp = MTPWithMLIP3(mlip_path=mlip_folder_path)
    # train
    mtp.train(
        train_structures=train_inputs.structure,
        train_energies=train_inputs.energy,
        train_forces=train_inputs.forces,
        train_stresses=None,
        max_dist=5,
        stress_weight=0,
        fitted_mtp_savedir=save_dir,
    )

    return mtp


def evaluate_mtp(eval_inputs: MTP_Inputs, mtp: MTPWithMLIP3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a trained MTP potential.

    Args:
        eval_inputs: inputs to evaluate. Should contain structure, energies and forces
        mtp: trained MTP potential.

    Returns:
       dataframe with original and predicted energies and forces.
    """
    # evaluate
    df_orig, df_predict = mtp.evaluate(
        test_structures=eval_inputs.structure,
        test_energies=eval_inputs.energy,
        test_forces=eval_inputs.forces,
        test_stresses=None,
    )
    return df_orig, df_predict


def get_metrics_from_pred(df_orig: pd.DataFrame, df_predict: pd.DataFrame) -> Tuple[float, float]:
    """Get mean absolute error on energy and forces from the outputs of MTP.

    Args:
        df_orig: dataframe with ground truth values
        df_predict: dataframe with MTP predictions

    Returns:
        MAE on energy in eV/atom and MAE on forces in eV/Å
    """
    # from demo in maml
    # get a single predicted energy per structure
    predicted_energy = df_predict.groupby('structure_index').agg({'energy': 'mean', 'atom_index': 'count'})
    # normalize by number of atoms
    predicted_energy = (predicted_energy['energy'] / predicted_energy['atom_index']).to_numpy()
    # same for ground truth
    gt_energy = df_orig.groupby('structure_index').agg({'energy': 'mean', 'atom_index': 'count'})
    gt_energy = (gt_energy['energy'] / gt_energy['atom_index']).to_numpy()

    predicted_forces = (df_predict[['fx', 'fy', 'fz']].to_numpy().flatten())
    gt_forces = (df_orig[['fx', 'fy', 'fz']].to_numpy().flatten())

    return mean_absolute_error(predicted_energy, gt_energy), mean_absolute_error(predicted_forces, gt_forces)


def main():
    """Train and evaluate an example for MTP."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--lammps_yaml', help='path to LAMMPS yaml file', required=True, nargs='+')
    parser.add_argument('--lammps_thermo', help='path to LAMMPS thermo output', required=True, nargs='+')
    parser.add_argument('--mlip_dir', help='directory to MLIP compilation folder', required=True)
    parser.add_argument('--output_dir', help='path to folder where outputs will be saved', required=True)
    args = parser.parse_args()

    lammps_yaml = args.lammps_yaml
    lammps_thermo_yaml = args.lammps_thermo
    assert len(lammps_yaml) == len(lammps_thermo_yaml), "LAMMPS outputs yaml should match thermodynamics output."

    mtp_inputs = prepare_mtp_inputs_from_lammps(lammps_yaml, lammps_thermo_yaml, atom_dict)
    mtp = train_mtp(mtp_inputs, args.mlip_dir, args.output_dir)
    print("Training is done")
    df_orig, df_predict = evaluate_mtp(mtp_inputs, mtp)
    energy_mae, force_mae = get_metrics_from_pred(df_orig, df_predict)
    print(f"MAE of training energy prediction is {energy_mae * 1000} meV/atom")
    print(f"MAE of training force prediction is {force_mae} eV/Å")


if __name__ == '__main__':
    main()

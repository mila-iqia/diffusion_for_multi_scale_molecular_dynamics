"""Script to train and evaluate a MTP.

Running the main() runs a debugging example. Entry points are train_mtp.
"""
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Structure
from sklearn.metrics import mean_absolute_error

from crystal_diffusion.models.mtp import MTPWithMLIP3
from crystal_diffusion.mlip.mtp_utils import prepare_mtp_inputs_from_lammps, crawl_lammps_directory, MTP_Inputs

atom_dict = {1: 'Si'}


def prepare_dataset(root_data_dir: str, atom_dict: Dict[int, str], mode: str = "train") -> MTP_Inputs:
    lammps_outputs, thermo_outputs = crawl_lammps_directory(root_data_dir, mode)
    mtp_dataset = prepare_mtp_inputs_from_lammps(lammps_outputs, thermo_outputs, atom_dict)
    return mtp_dataset


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

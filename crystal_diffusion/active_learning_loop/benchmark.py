import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import yaml
from hydra.utils import instantiate

from crystal_diffusion.mlip.mtp_train import train_mtp, prepare_dataset, evaluate_mtp
from crystal_diffusion.models.mtp import MTPWithMLIP3


def get_arguments() -> argparse.Namespace:
    """Parse arguments.

    Returns:
        args: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtp_config', help='path to data directory', required=True)
    args = parser.parse_args()
    return args


@dataclass(kw_only=True)
class MTPArguments:
    training_data_dir: str  # training data directory
    evaluation_data_dir: str  # evaluation data directory
    mlip_dir: str  # directory with the mlp executable
    output_dir: str  # directory where to save the MTP outputs


@dataclass(kw_only=True)
class StructureEvaluationArguments:
    evaluation_criteria: str ='nbh_grades'
    criteria_threshold: float = 10
    number_of_structures: int = None
    extraction_radius: float = 3


def train_mlip(mtp_args: MTPArguments, atom_dict: Dict[int, str]) -> MTPWithMLIP3:
    """Train a MTP model using the specified data.

    Args:
        mtp_args: MTPArguments data class
        atom_dict: map between atom names and indices used by LAMMPS

    Returns:
        trained MTP model
    """
    # TODO make it more configurable
    train_datasets = prepare_dataset(mtp_args.training_data_dir, atom_dict, mode="train")
    # create the output directory if it doesn't exist already
    trained_mtp = train_mtp(train_datasets, mlip_folder_path=mtp_args.mlip_dir, save_dir=mtp_args.output_dir)
    return trained_mtp


def evaluate_mlip(mtp_args: MTPArguments, atom_dict: Dict[int, str], mtp: MTPWithMLIP3) -> pd.DataFrame:
    """Evaluate a MTP model using the specified data.

    Args:
        mtp_args: MTPArguments data class
        atom_dict:  map between atom names and indices used by LAMMPS
        mtp: trained MTP model

    Returns:
        dataframe with a column specified the structure, a column with the atom index, 3 columns with the x,y,z
        coordinates and a column with the MaxVol criteria (nbh_grades)
    """
    evaluation_datasets = prepare_dataset(mtp_args.evaluation_data_dir, atom_dict, mode="evaluation")
    # TODO the current evaluation method also returns the ground truth informations - this won't always be the case
    # TODO make more configurable
    _, prediction_df = evaluate_mtp(evaluation_datasets, mtp)
    return prediction_df


def get_structures_for_retraining(prediction_df: pd.DataFrame,
                                  criteria_threshold: Optional[float] = None,
                                  number_of_structures: Optional[int] = None,
                                  evaluation_criteria: str = 'nbh_grades',
                                  structure_index: str = 'structure_index'
                                  ) -> List[pd.DataFrame]:
    assert criteria_threshold is not None or number_of_structures is not None, \
        "criteria_threshold or number_of_structures should be set."
    # get the highest evaluation_criteria for each structure i.e. only the worst atom counts for structure selection
    criteria_by_structure = prediction_df[[evaluation_criteria, structure_index]].groupby(structure_index).max()
    # find the top number_of_structures
    structures_indices = criteria_by_structure.sort_values(by=evaluation_criteria, ascending=False)
    if number_of_structures is not None:
        structures_indices = structures_indices[:number_of_structures]
    else:  #  criteria_threshold is not None
        structures_indices = structures_indices[structures_indices[evaluation_criteria] >= criteria_threshold]
    structures_indices = structures_indices.index.to_list()
    assert len(structures_indices) > 0, "No structure meet the criteria."
    structures_to_retrain = []
    for idx in structures_indices:
        structures_to_retrain.append(prediction_df[prediction_df[structure_index] == idx])
    return structures_to_retrain


def extract_target_region(structure_df: pd.DataFrame,
                          extraction_radius: float,
                          evaluation_criteria: str ='nbh_grades') -> pd.DataFrame:
    """Extract the atom with the worst evaluation criteria and all the atoms within a distance extraction_radious.

    Args:
        structure_df: dataframe with the atomic positions and the evaluation criteria (e.g. MaxVol value)
        extraction_radius: include all atoms within this distance of the targeted atom
        evaluation_criteria: name of the evaluation criteria. Defaults to nbh_grades (maxvol in MTP)

    Returns:
        dataframe with the atomic coordinates in columns x, y, z
    """
    # extract the worst ato and a region around of radius extraction_radius
    # TODO better method to determine radius: number of atoms ?
    target_atom = structure_df[evaluation_criteria].idxmax()
    target_position = structure_df.loc[target_atom][['x', 'y', 'z']]
    structure_df.loc[:, 'distance_squared'] = structure_df.apply(
        lambda x: sum([(x[i] - target_position[i]) ** 2 for i in ['x', 'y', 'z']]), axis=1)
    atom_positions = structure_df.loc[structure_df['distance_squared'] <= extraction_radius ** 2, ['x', 'y', 'z']]
    return atom_positions


def generate_new_structures(fixed_atoms: List[pd.DataFrame]) -> None:
    pass


def main():
    # args = get_arguments()
    # TODO get mtp_config_path from the args
    mtp_config_path = "/Users/simonb/ic-collab/courtois_collab/crystal_diffusion/experiments/active_learning_benchmark/"
    mtp_config_path = os.path.join(mtp_config_path, "config", "mtp_training.yaml")
    with open(mtp_config_path, 'r') as stream:
        mtp_config = yaml.load(stream, Loader=yaml.FullLoader)
    # use hydra to convert the yaml file in a dataclass format
    mtp_config = instantiate(mtp_config)
    os.makedirs(mtp_config.output_dir, exist_ok=True)
    atom_dict = {1: "Si"}  # TODO this should be define somewhere smart
    # STEP 1: train a MLIP
    trained_mtp = train_mlip(mtp_config, atom_dict)
    # STEP 2: evaluate the MLIP
    prediction_df = evaluate_mlip(mtp_config, atom_dict, trained_mtp)
    # STEP 3: identify the problematic structures
    # TODO extraction_params should come from a config file with hydra instantiate
    extraction_params = StructureEvaluationArguments()
    structures_to_retrain = get_structures_for_retraining(prediction_df,
                                                          criteria_threshold=extraction_params.criteria_threshold,
                                                          number_of_structures=extraction_params.number_of_structures,
                                                          evaluation_criteria=extraction_params.evaluation_criteria)
    # STEP 4: extract the region
    bad_regions = [extract_target_region(s, extraction_radius=extraction_params.extraction_radius)
                   for s in structures_to_retrain]
    # STEP 5: call the generative model to create new candidates

    print('hello')


if __name__ == '__main__':
    main()

import argparse
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from hydra.utils import instantiate

from crystal_diffusion.active_learning_loop.utils import (
    extract_target_region, get_structures_for_retraining)
from crystal_diffusion.models.mlip.mtp import MTPWithMLIP3


class ActiveLearningLoop:
    """Method to train, evaluate and fine-tune a MLIP."""
    def __init__(self,
                 meta_config: str,
                 ):
        """Active learning benchmark.

        Includes methods to train & evaluate a MLIP, isolate bad sub-structures, repaint new structures and retrain
        the MLIP.

        Args:
            meta_config: path to a yaml configuration with the parameters for the modules in the class
        """
        assert os.path.exists(meta_config), "configuration file for active learning loop does not exist."
        # define the modules in the __init__ function
        self.data_paths, self.mlip_model, self.eval_config, self.structure_generation = None, None, None, None
        self.oracle = None
        # use hydra to convert the yaml into modules and other data classes
        self.parse_config(meta_config)
        self.atom_dict = {1: "Si"}  # TODO this should be define somewhere smart
        self.trained_mlips = []  # history of trained MLIPs (optional - not sure if we should keep this)
        self.training_sets = []  # history of training sets

    def parse_config(self, meta_config: str):
        """Read a configuration file and instantiate the different blocks with hydra.

        The configuration file should have the following blocks of parameters:
            active_learning_data: dataset paths
            mlip_model: MLIP module training parameters
            structure_evaluation: identification and isolation of the atomic regions to finetune the MLIP

        Args:
            meta_config: path to configuration yaml file
        """
        with open(meta_config, 'r') as stream:
            meta_config = yaml.load(stream, Loader=yaml.FullLoader)
        # paths to the training & evaluation datasets
        self.data_paths = instantiate(meta_config['active_learning_data'])
        # MLIP model - for example MTP
        self.mlip_model = instantiate(meta_config['mlip'])
        # parameters to find and isolate the problematic regions in the evaluation dataset
        self.eval_config = instantiate(meta_config['structure_evaluation'])
        # structure generation module
        self.structure_generation = instantiate(meta_config['repainting_model'])
        # force labeling module
        self.oracle = instantiate(meta_config['oracle'])

    def train_mlip(self, round: int = 1, training_set: Optional[Any] = None) -> str:
        """Train a MLIP using the parameters specified in the configuration file.

        Args:
            round (optional): current round of training. Used to track now configurations in the training set. A round
                includes the initial training and the evaluation process.
            training_set (optional): if specified, use this dataset for training. Otherwise, use the dataset from the
               paths in the configuration file. Defaults to None.

        Returns:
            path to the trained MLIP model
        """
        if training_set is None:
            if len(self.training_sets) == 0:
                self.training_sets = [self.mlip_model.prepare_dataset_from_lammps(
                    root_data_dir=self.data_paths.training_data_dir,
                    atom_dict=self.atom_dict,
                    mode="train"
                )]
            training_set = self.mlip_model.merge_inputs(self.training_sets)

        trained_mtp = self.mlip_model.train(training_set, mlip_name=f'mlip_round_{round}')
        self.trained_mlips.append(trained_mtp)  # history of trained MLIPs ... not sure if useful
        return trained_mtp

    def evaluate_mlip(self, round: int = 1, mlip_name: Optional[str] = None, forces_available: bool = True
                      ) -> pd.DataFrame:
        """Evaluate a MLIP using the parameters specified in the configuration file.

        Args:
            round (optional): current round of training. Defaults to 1.
            mlip_name (optional): if not None, use this MTP to evaluate the dataset.
            forces_available (optional): if True, get the ground truth forces from the dataset.

        Returns:
            dataframe with the atomic indices, positions, forces and evaluation criteria
        """
        evaluation_dataset = self.mlip_model.prepare_dataset_from_lammps(
            root_data_dir=self.data_paths.evaluation_data_dir,
            atom_dict=self.atom_dict,
            mode="evaluation",
            get_forces=forces_available
        )
        # first returned element is the ground truth DF
        # TODO make sure this works even if the GT is not available...
        if mlip_name is None:
            mlip_name = os.path.join(self.mlip_model.savedir, f'mlip_round_{round}.almtp')
        _, prediction_df = self.mlip_model.evaluate(evaluation_dataset, mlip_name=mlip_name)

        return prediction_df

    def get_bad_structures(self, prediction_df: pd.DataFrame) -> List[pd.DataFrame]:
        """Find the structures with a high uncertainty based on the configuration file parameters.

        Args:
            prediction_df: evaluation outputs of the MLIP model. Should contain atomic positions, uncertainty criteria
               and structure indices.

        Returns:
            list of structures with a high uncertainty criteria.
        """
        num_structures = self.eval_config.number_of_structures
        structures_to_retrain = get_structures_for_retraining(prediction_df,
                                                              criteria_threshold=self.eval_config.criteria_threshold,
                                                              number_of_structures=num_structures,
                                                              evaluation_criteria=self.eval_config.evaluation_criteria)
        return structures_to_retrain

    def excise_worst_atom(self, structures_to_retrain: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """For a given structure, isolate the atom with the highest uncertainty criteria.

        Args:
            structures_to_retrain: list of dataframes with the atomic positions and evaluate criteria

        Returns:
            list of dataframes with only the targeted region
        """
        # we assume the extraction region to be a sphere of radius extraction_radius around the worst atoms
        # if more than 1 atom are bad in a structure, we only extract the worst
        # TODO implement other extraction methods
        bad_regions = [extract_target_region(s,
                                             extraction_radius=self.eval_config.extraction_radius,
                                             evaluation_criteria=self.eval_config.evaluation_criteria)
                       for s in structures_to_retrain]
        return bad_regions

    def get_structure_candidate_from_generative_model(self,
                                                      fixed_atoms: pd.DataFrame,
                                                      number_of_candidates: int = 1
                                                      ) -> pd.DataFrame:
        """Generate new structures around the specified fixed atoms.

        Args:
            fixed_atoms: dataframe with the atom type, coordinates and unit cell information
            number_of_candidates: how many structure to generate. Defaults to 1.

        Returns:
            dataframe with the atom type, coordinates and unit cell

        """
        # TODO: call the diffusion model and get number_of_candidates samples with repaint using the fixed_atoms
        if self.structure_generation.model == 'dev_dummy':  # replace with a wrapper around the diffusion model
            # and hydra instantiate
            return fixed_atoms
        else:
            raise NotImplementedError('Only dev_dummy is supported at the moment.')

    def new_structure_to_csv(self, new_structures: List[pd.DataFrame], round: int = 1):
        """Save the generated structures in a csv format in the output dir.

        Args:
            new_structures: structures proposed by the generative model
            round: current round of training. Defaults to 1.
        """
        root_data_dir = os.path.join(self.data_paths.output_dir, f'new_structures_round_{round}')
        os.makedirs(root_data_dir, exist_ok=True)
        for i, new_struc in enumerate(new_structures):
            new_struc.to_csv(os.path.join(root_data_dir, f'structure_{i}.csv'), index=False)

    def get_labels_from_oracle(self, round: int = 1) -> Any:
        """Compute energy and forces from an oracle such as LAMMPS for the new candidates generated in a round of AL.

        Args:
            round (optional): round of retraining. Defaults to 1.

        Returns:
            mlip data input (for example, MTPInputs)
        """
        new_labeled_samples = []
        for file in os.listdir(os.path.join(self.data_paths.output_dir, f'new_structures_round_{round}')):
            if file.endswith('.csv'):
                new_labeled_samples.append(self.call_oracle(
                    os.path.join(self.data_paths.output_dir, f'new_structures_round_{round}', file)
                ))
        new_labeled_samples = self.mlip_model.merge_inputs(new_labeled_samples)
        return new_labeled_samples

    def call_oracle(self, path_to_file: str) -> Any:
        """Compute energy and forces for a given atomic structure.

        Args:
            path_to_file: path to csv file containing the atomic positions and structure information

        Returns:
            mlip data inputs (for example, MTPInputs)
        """
        data = pd.read_csv(path_to_file)
        cartesian_positions = data[['x', 'y', 'z']].to_numpy()
        box = np.eye(3, 3) * 5.43  # TODO this is bad - fix this
        atom_type = np.ones(cartesian_positions.shape[0], dtype=np.integer)  # TODO also bad
        energy, forces = self.oracle(cartesian_positions, box, atom_type)
        labels_as_mtp = self.mlip_model.prepare_dataset_from_numpy(
            cartesian_positions,
            box,
            forces,
            energy,
            atom_type,
        )
        return labels_as_mtp

    def round_of_active_learning_loop(self, trained_mlip: Optional[MTPWithMLIP3] = None
                                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Do a full loop of activate learning.

        The following steps are done in sequence:
            - train a MLIP from the training set specified in the config file if trained_mlip is not specified
            - evaluate the MLIP with the evaluation set specified in the config file
            - find the "bad" structures in the evaluation set based on the criteria from the config file
            - excise the problematic regions
            - generate new candidates based on these regions
            - call the oracle to get the labels for the new generated candidates
            - retrain the MLIP
            - evaluate the MLIP again

        Args:
            trained_mlip (optional): if not None, use this MLIP as a starting point. If None, train a MLIP from scratch
                using the training data specified in the config file.

        Returns:
            dataframe with the MLIP evaluation results before finetuning with the generated structures
            dataframe with the MLIP evaluation results after finetuning with the generated structures
        """
        # one round from a known mtp (or train from provided training set)
        # evaluate, find candidates and update MTP
        # return the updated MTP
        if trained_mlip is None:
            trained_mlip = self.train_mlip()
        pred_df = self.evaluate_mlip(mlip_name=trained_mlip)
        bad_structures = self.get_bad_structures(pred_df)
        bad_regions = self.excise_worst_atom(bad_structures)
        new_candidates = [self.get_structure_candidate_from_generative_model(x) for x in bad_regions]
        self.new_structure_to_csv(new_candidates)
        new_labeled_candidates = self.get_labels_from_oracle()
        new_training_set = self.mlip_model.merge_inputs([self.training_sets[-1], new_labeled_candidates])
        self.training_sets.append(new_training_set)
        new_mtp = self.train_mlip()
        new_pred_df = self.evaluate_mlip(mlip_name=new_mtp)
        return pred_df, new_pred_df

    def evaluate_mtp_update(self, original_predictions: pd.DataFrame, updated_predictions) -> Tuple[float, float]:
        """Find the evaluation criteria in the original predictions and the corresponding value after retraining.

        Args:
            original_predictions: MLIP predictions before retraining
            updated_predictions: MLIP predictions after retraining

        Returns:
             worst evaluation_criteria (e.g. MaxVol) in the original evaluation
             corresponding value after retraining with new samples. Not guaranteed to be the maximum value.
        """
        # find the highest MaxVol in the original predictions - identified by the atom index and structure index
        # TODO we assume a max - but it could be a min i
        criteria = self.eval_config.evaluation_criteria
        atom_index, structure_index, original_value = original_predictions.iloc[
            original_predictions[criteria].argmax()][['atom_index', 'structure_index', criteria]]
        updated_value = updated_predictions.loc[
            (updated_predictions['atom_index'] == atom_index)
            & (updated_predictions['structure_index'] == structure_index), criteria].values.item()
        return original_value, updated_value


def get_arguments() -> argparse.Namespace:
    """Parse arguments.

    Returns:
        args: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to data directory', required=True)
    args = parser.parse_args()
    return args


def main():
    """Example to do an active learning loop once."""
    args = get_arguments()
    # TODO get mtp_config_path from the args
    config_path = args.config
    al_loop = ActiveLearningLoop(config_path)
    initial_df, new_df = al_loop.round_of_active_learning_loop()
    al_loop.evaluate_mtp_update(initial_df, new_df)


if __name__ == '__main__':
    main()

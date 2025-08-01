"""Pretrain Flare.

This script generates many FLARE "checkpoints" by successively adding more data
from the training dataset to a FLARE model. The hyperparameter sigma is varied to generate
multiple checkpoints for HP tuning.
"""

import logging
import pickle

import numpy as np
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import (
    FlareConfiguration, FlareTrainer)

experiment_dir = TOP_DIR / "experiments/active_learning/pretraining_flare/"
data_dir = experiment_dir / "data"

checkpoint_top_dir = experiment_dir / "flare_checkpoints"
checkpoint_top_dir.mkdir(parents=True, exist_ok=True)

element_list = ["Si"]
variance_type = "local"
seed = 42

logging.basicConfig(level=logging.INFO)

list_sigma_ef = 0.1 ** np.arange(6)

sigma = 1.0

# Train FLARE with a variable number of structures.
list_number_of_training_structures = np.arange(1, 17)

if __name__ == "__main__":
    np.random.seed(seed)

    logging.info("Loading data")
    with open(data_dir / "train_labelled_structures.pkl", "rb") as fd:
        list_train_labelled_structures = pickle.load(fd)

    logging.info("Instantiate FLARE models")
    flare_trainer_dict = dict()
    for sigma_ef in tqdm(list_sigma_ef, "SIGMA"):
        flare_configuration = FlareConfiguration(
            cutoff=5.0,
            elements=element_list,
            n_radial=12,
            lmax=3,
            initial_sigma=sigma,
            initial_sigma_e=sigma_ef,
            initial_sigma_f=sigma_ef,
            initial_sigma_s=1.0,
            variance_type="local",
        )
        flare_trainer_dict[sigma_ef] = FlareTrainer(flare_configuration)

    logging.info("Adding training data to various FLARE models.")
    for number_of_training_structures in tqdm(list_number_of_training_structures, "N"):
        idx = number_of_training_structures - 1
        labelled_structure = list_train_labelled_structures[idx]

        # Add a random subset of environments, using exactly the same for all sigma.
        number_of_atoms = len(labelled_structure.structure)
        active_environment_indices = list(np.random.randint(0, number_of_atoms, (8,)))

        for sigma_level, sigma_ef in tqdm(enumerate(list_sigma_ef, 1), "SIGMA"):
            flare_trainer = flare_trainer_dict[sigma_ef]
            flare_trainer.add_labelled_structure(
                labelled_structure,
                active_environment_indices=active_environment_indices,
            )

            checkpoint_dir = (
                checkpoint_top_dir / f"sigma_level_{sigma_level}_n_{number_of_training_structures}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=False)

            checkpoint_path = checkpoint_dir / "flare_model_pretrained.json"
            flare_trainer.write_checkpoint_to_disk(checkpoint_path)
            pair_coeff_file_path, mapped_uncertainty_file_path = (
                flare_trainer.write_mapped_model_to_disk(
                    checkpoint_dir, version="_pretrained"
                )
            )

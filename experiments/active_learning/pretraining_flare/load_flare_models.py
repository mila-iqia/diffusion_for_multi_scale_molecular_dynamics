"""Load Flare Models.

This script generates many FLARE "checkpoints" by successively adding more data
from the training dataset to a FLARE model. The various hyperparameters are set to
arbitrary values since they can easily be changed as needed.
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

# Pick arbitrary values
sigma = 1.0
sigma_e = 1.0
sigma_f = 1.0
sigma_s = 1.0

# Train FLARE with a variable number of structures.
list_number_of_training_structures = np.arange(1, 17)

if __name__ == "__main__":
    np.random.seed(seed)

    logging.info("Loading data")
    with open(data_dir / "train_labelled_structures.pkl", "rb") as fd:
        list_train_labelled_structures = pickle.load(fd)

    logging.info("Instantiate FLARE models")
    flare_configuration = FlareConfiguration(
        cutoff=5.0,
        elements=element_list,
        n_radial=12,
        lmax=3,
        initial_sigma=sigma,
        initial_sigma_e=sigma_e,
        initial_sigma_f=sigma_f,
        initial_sigma_s=sigma_s,
        variance_type="local",
    )
    flare_trainer = FlareTrainer(flare_configuration)

    logging.info("Adding training data to various FLARE models.")
    for number_of_training_structures in tqdm(list_number_of_training_structures, "N"):
        idx = number_of_training_structures - 1
        labelled_structure = list_train_labelled_structures[idx]

        # Add a random subset of environments
        number_of_atoms = len(labelled_structure.structure)
        active_environment_indices = list(np.random.randint(0, number_of_atoms, (8,)))

        flare_trainer.add_labelled_structure(
            labelled_structure,
            active_environment_indices=active_environment_indices,
        )

        checkpoint_dir = (
            checkpoint_top_dir / f"number_of_structures_{number_of_training_structures}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=False)

        checkpoint_path = checkpoint_dir / "flare_model_preloaded.json"
        flare_trainer.write_checkpoint_to_disk(checkpoint_path)

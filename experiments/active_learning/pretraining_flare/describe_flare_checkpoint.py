"""Pretrain Flare.

This script generates many FLARE "checkpoints" by successively adding more data
from the training dataset to a FLARE model. The hyperparameter sigma is varied to generate
multiple checkpoints for HP tuning.
"""

import logging

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import \
    FlareTrainer

experiment_dir = TOP_DIR / "experiments/active_learning/pretraining_flare/"
checkpoint_top_dir = experiment_dir / "flare_checkpoints" / "pretrained_flare"
checkpoint_path = checkpoint_top_dir / "flare_model_pretrained.json"

element_list = ["Si"]
variance_type = "local"
seed = 42

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    flare_trainer = FlareTrainer.from_checkpoint(checkpoint_path)

    sparse_gp = flare_trainer.sgp_model.sparse_gp

    number_of_training_structures = len(sparse_gp.training_structures)

    number_of_sparse_points, number_of_labels = sparse_gp.Kuf.shape

    sigma, sigma_e, sigma_f, sigma_s = sparse_gp.hyperparameters

    print(f"number of training structures : {number_of_training_structures}")
    print(f"      number of sparse points : {number_of_sparse_points}")
    print(f"                number labels : {number_of_labels}")
    print(f"                        sigma : {sigma}")
    print(f"                      sigma_e : {sigma_e}")
    print(f"                      sigma_f : {sigma_f}")
    print(f"                      sigma_s : {sigma_s}")

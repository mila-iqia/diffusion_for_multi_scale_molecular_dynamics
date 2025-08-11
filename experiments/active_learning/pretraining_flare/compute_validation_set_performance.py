import glob
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.flare_single_point_calculator import \
    FlareSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.utils import \
    compute_errors_and_uncertainties
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import \
    FlareTrainer

logging.basicConfig(level=logging.INFO)

experiment_dir = TOP_DIR / "experiments/active_learning/pretraining_flare/"

data_dir = experiment_dir / "data"
checkpoint_top_dir = experiment_dir / "flare_checkpoints"

output_dir = experiment_dir / "validation_performance"
output_dir.mkdir(parents=True, exist_ok=True)

sigma = 1.0
# scanning  over a range of values.
list_sigma_e = np.array([1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])
list_sigma_f = np.array([1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])

if __name__ == "__main__":

    with open(data_dir / "valid_labelled_structures.pkl", "rb") as fd:
        list_valid_labelled_structures = pickle.load(fd)

    checkpoint_directories = glob.glob(str(checkpoint_top_dir / "number_of_structures_*"))
    number_of_directories = len(checkpoint_directories)

    list_rows = []
    for idx, checkpoint_directory in enumerate(checkpoint_directories, 1):
        logging.info(
            f"Processing {checkpoint_directory} ({idx} of {number_of_directories})"
        )
        checkpoint_path = Path(checkpoint_directory) / "flare_model_preloaded.json"

        flare_trainer = FlareTrainer.from_checkpoint(checkpoint_path)
        number_of_structures = flare_trainer.sgp_model.sparse_gp.n_energy_labels

        for sigma_e in list_sigma_e:
            for sigma_f in list_sigma_f:
                hyperparameters = np.array([sigma, sigma_e, sigma_f, 1.0])
                flare_trainer.sgp_model.sparse_gp.set_hyperparameters(hyperparameters)
                flare_trainer.sgp_model.sparse_gp.precompute_KnK()

                flare_calculator = FlareSinglePointCalculator(sgp_model=flare_trainer.sgp_model)

                row = dict(sigma=sigma, sigma_e=sigma_e, sigma_f=sigma_f,
                           number_of_structures=number_of_structures)

                flare_results = compute_errors_and_uncertainties(
                    flare_calculator, list_valid_labelled_structures
                )

                row.update(flare_results)
                list_rows.append(row)

    df = pd.DataFrame(list_rows)
    df.to_pickle(output_dir / "validation_set_performance.pkl")

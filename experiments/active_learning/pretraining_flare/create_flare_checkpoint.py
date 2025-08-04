import logging
import pickle
from pathlib import Path

import numpy as np

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.lammps_runner import \
    LammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.flare_single_point_calculator import \
    FlareSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.mapped_flare_single_point_calculator import \
    MappedFlareSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.utils import \
    compute_errors_and_uncertainties
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import \
    FlareTrainer

logging.basicConfig(level=logging.INFO)

lammps_executable_path = Path("/Users/brunorousseau/sources/lammps/build/lmp")

experiment_dir = TOP_DIR / "experiments/active_learning/pretraining_flare/"

output_dir = experiment_dir / "validation_performance"

data_dir = experiment_dir / "data"
checkpoint_top_dir = experiment_dir / "flare_checkpoints"

pretrained_checkpoint_dir = checkpoint_top_dir / "pretrained_flare"
pretrained_checkpoint_dir.mkdir(parents=True, exist_ok=True)

sigma = 1.0
sigma_e = 0.1
sigma_f = 0.001
number_of_structures = 16

if __name__ == "__main__":
    logging.info(f"Loading the {number_of_structures} atom pre-loaded FLARE")
    preloaded_flare_path = (checkpoint_top_dir
                            / f"number_of_structures_{number_of_structures}" / "flare_model_preloaded.json")
    flare_trainer = FlareTrainer.from_checkpoint(preloaded_flare_path)

    logging.info("Setting the correct HPs")
    hyperparameters = np.array([sigma, sigma_e, sigma_f, 1.0])
    flare_trainer.sgp_model.sparse_gp.set_hyperparameters(hyperparameters)
    flare_trainer.sgp_model.sparse_gp.precompute_KnK()

    logging.info("Writing pretrained checkpoint to file")
    checkpoint_path = pretrained_checkpoint_dir / "flare_model_pretrained.json"
    flare_trainer.write_checkpoint_to_disk(checkpoint_path)
    pair_coeff_file_path, mapped_uncertainty_file_path = (
        flare_trainer.write_mapped_model_to_disk(pretrained_checkpoint_dir, version="_pretrained"))

    logging.info("Instantiate calculators")
    flare_calculator = FlareSinglePointCalculator(sgp_model=flare_trainer.sgp_model)

    lammps_runner = LammpsRunner(
        lammps_executable_path, mpi_processors=4, openmp_threads=4
    )

    mapped_flare_calculator = MappedFlareSinglePointCalculator(
        lammps_runner=lammps_runner,
        pair_coeff_file_path=pair_coeff_file_path,
        mapped_uncertainty_file_path=mapped_uncertainty_file_path,
    )

    logging.info("Load test dataset")
    with open(data_dir / "test_labelled_structures.pkl", "rb") as fd:
        list_test_labelled_structures = pickle.load(fd)

    logging.info("Compute test dataset errors and uncertainties")

    results = dict(sigma=sigma, sigma_e=sigma_e, sigma_f=sigma_f, number_of_structures=number_of_structures)

    flare_results = compute_errors_and_uncertainties(flare_calculator, list_test_labelled_structures)

    for key, value in flare_results.items():
        results[f"flare_{key}"] = value

    mapped_flare_results = compute_errors_and_uncertainties(
        mapped_flare_calculator, list_test_labelled_structures
    )

    for key, value in mapped_flare_results.items():
        results[f"mapped_flare_{key}"] = value

    with open(output_dir / "test_set_performance.pkl", "wb") as fd:
        pickle.dump(results, fd)

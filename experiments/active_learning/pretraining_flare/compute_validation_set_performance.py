import glob
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.lammps_runner import \
    LammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.flare_single_point_calculator import \
    FlareSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.mapped_flare_single_point_calculator import \
    MappedFlareSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import \
    FlareTrainer


def compute_errors_and_uncertainties(single_point_calculator, list_labelled_structures):
    """Compute errors and uncertainties over a given dataset with a given single point calculator."""
    # All values for all atoms
    list_all_force_errors = []
    list_all_uncertainties = []

    # Aggregated values per labelled_structure
    list_force_rmse_per_structure = []
    list_energy_errors_per_structure = []

    for labelled_structure in list_labelled_structures:
        result = single_point_calculator.calculate(
            structure=labelled_structure.structure
        )

        force_errors = np.linalg.norm(result.forces - labelled_structure.forces, axis=1)
        list_all_force_errors.append(force_errors)

        list_all_uncertainties.append(result.uncertainties)

        force_rmse = np.sqrt(np.mean(force_errors**2))
        list_force_rmse_per_structure.append(force_rmse)

        energy_error = result.energy - labelled_structure.energy
        list_energy_errors_per_structure.append(energy_error)

    list_all_force_errors = np.concatenate(list_all_force_errors)
    list_all_uncertainties = np.concatenate(list_all_uncertainties)

    list_force_rmse_per_structure = np.array(list_force_rmse_per_structure)
    list_energy_errors_per_structure = np.array(list_energy_errors_per_structure)

    mean_force_rmse = np.mean(list_force_rmse_per_structure)
    energy_rmse = np.sqrt(np.mean(np.array(list_energy_errors_per_structure) ** 2))

    results = dict(
        all_force_errors=list_all_force_errors,
        all_uncertainties=list_all_uncertainties,
        force_rmse_per_structure=list_force_rmse_per_structure,
        energy_error_per_structure=list_energy_errors_per_structure,
        mean_force_rmse=mean_force_rmse,
        energy_rmse=energy_rmse,
    )

    return results


logging.basicConfig(level=logging.INFO)

lammps_executable_path = Path("/Users/brunorousseau/sources/lammps/build/lmp")

experiment_dir = TOP_DIR / "experiments/active_learning/pretraining_flare/"

data_dir = experiment_dir / "data"
checkpoint_top_dir = experiment_dir / "flare_checkpoints"

output_dir = experiment_dir / "validation_performance"
output_dir.mkdir(parents=True, exist_ok=True)

# the MAPPED flare is SLOW, probably because of yaml parsing.
# We'll only compute a subset.
list_n_for_mapped_flare_calculations = [5, 10, 15]

if __name__ == "__main__":
    lammps_runner = LammpsRunner(
        lammps_executable_path, mpi_processors=4, openmp_threads=4
    )

    with open(data_dir / "valid_labelled_structures.pkl", "rb") as fd:
        list_valid_labelled_structures = pickle.load(fd)

    checkpoint_directories = glob.glob(str(checkpoint_top_dir / "sigma_*_n_*"))
    number_of_directories = len(checkpoint_directories)

    list_rows = []
    for idx, checkpoint_directory in enumerate(checkpoint_directories, 1):
        logging.info(
            f"Processing {checkpoint_directory} ({idx} of {number_of_directories})"
        )
        checkpoint_path = Path(checkpoint_directory) / "flare_model_pretrained.json"

        flare_trainer = FlareTrainer.from_checkpoint(checkpoint_path)
        flare_calculator = FlareSinglePointCalculator(sgp_model=flare_trainer.sgp_model)

        sigma = flare_trainer.flare_configuration.initial_sigma
        number_of_structures = flare_trainer.sgp_model.sparse_gp.n_energy_labels

        row = dict(sigma=sigma, number_of_structures=number_of_structures)

        flare_results = compute_errors_and_uncertainties(
            flare_calculator, list_valid_labelled_structures
        )

        for key, value in flare_results.items():
            row[f"flare_{key}"] = value

        if number_of_structures in list_n_for_mapped_flare_calculations:
            pair_coeff_file_path = Path(checkpoint_directory) / "lmp_pretrained.flare"
            mapped_uncertainty_file_path = (
                Path(checkpoint_directory) / "map_unc_lmp_pretrained.flare"
            )
            mapped_flare_calculator = MappedFlareSinglePointCalculator(
                lammps_runner=lammps_runner,
                pair_coeff_file_path=pair_coeff_file_path,
                mapped_uncertainty_file_path=mapped_uncertainty_file_path,
            )

            mapped_flare_results = compute_errors_and_uncertainties(
                mapped_flare_calculator, list_valid_labelled_structures
            )

            for key, value in mapped_flare_results.items():
                row[f"mapped_flare_{key}"] = value

        list_rows.append(row)

    df = pd.DataFrame(list_rows)
    df.to_pickle(output_dir / "validation_set_performance.pkl")

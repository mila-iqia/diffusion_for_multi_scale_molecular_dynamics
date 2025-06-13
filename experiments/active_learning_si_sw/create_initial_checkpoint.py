import argparse
from pathlib import Path

import numpy as np
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics import DATA_DIR, TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.lammps_runner import \
    LammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.stillinger_weber_single_point_calculator import \
    StillingerWeberSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import (
    FlareConfiguration, FlareTrainer)

experiment_dir = TOP_DIR / "experiments/active_learning_si_sw/"
sw_coefficients_file_path = DATA_DIR / "stillinger_weber_coefficients/Si.sw"

reference_directory = experiment_dir / "reference"

element_list = ["Si"]

variance_type = "local"

flare_configuration = FlareConfiguration(
    cutoff=5.0,
    elements=element_list,
    n_radial=12,
    lmax=3,
    initial_sigma=1000.0,
    initial_sigma_e=1.0,
    initial_sigma_f=0.050,
    initial_sigma_s=1.0,
    variance_type="local",
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_lammps_executable",
        help="path to a LAMMPS executable that is compatible with ARTn and FLARE.",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--path_to_initial_flare_checkpoint",
        help="Where should the FLARE model checkpoint be written.",
        default=None,
        required=True,
    )
    args = parser.parse_args()

    lammps_runner = LammpsRunner(
        lammps_executable_path=Path(args.path_to_lammps_executable),
        mpi_processors=4,
        openmp_threads=4,
    )

    oracle_calculator = StillingerWeberSinglePointCalculator(
        lammps_runner, sw_coefficients_file_path
    )

    flare_trainer = FlareTrainer(flare_configuration)

    # Add a single environment to the flare trainer
    # Add a random subset of environments from the 0th structure.
    initial_configuration_file_path = reference_directory / "initial_configuration.dat"
    initial_structure = LammpsData.from_file(
        str(initial_configuration_file_path), atom_style="atomic", sort_id=True
    ).structure
    number_of_atoms = len(initial_structure)
    random_indices = np.random.randint(0, number_of_atoms, (8,))

    labelled_structure = oracle_calculator.calculate(initial_structure)

    active_environment_indices = list(random_indices)

    flare_trainer.add_labelled_structure(
        labelled_structure, active_environment_indices=active_environment_indices
    )

    flare_trainer.write_checkpoint_to_disk(Path(args.path_to_initial_flare_checkpoint).absolute())

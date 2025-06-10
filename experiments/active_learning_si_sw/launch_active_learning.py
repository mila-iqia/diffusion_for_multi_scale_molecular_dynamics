from pathlib import Path

import numpy as np
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics import DATA_DIR, TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.active_learning import \
    ActiveLearning
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.dynamic_driver.artn_driver import \
    ArtnDriver
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.lammps_runner import \
    LammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.base_sample_maker import (
    NoOpSampleMaker, NoOpSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.stillinger_weber_single_point_calculator import \
    StillingerWeberSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_hyperparameter_optimizer import (
    FlareHyperparametersOptimizer, FlareOptimizerConfiguration)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import (
    FlareConfiguration, FlareTrainer)

# This will change on different machines.
sources_directory = Path("/Users/brunorousseau/sources/")

experiment_dir = TOP_DIR / "experiments/active_learning_si_sw"

sw_coefficients_file_path = DATA_DIR / "stillinger_weber_coefficients/Si.sw"

lammps_executable_path = sources_directory / "lammps/build/lmp"
artn_library_plugin_path = sources_directory / "artn-plugin/build/libartn.dylib"

reference_directory = experiment_dir / "reference"

element_list = ['Si']

list_uncertainty_thresholds = [0.1**pow for pow in np.arange(2, 5)]
list_campaign_ids = list(range(1, len(list_uncertainty_thresholds) + 1))

variance_type = 'local'

flare_configuration = FlareConfiguration(cutoff=5.0,
                                         elements=element_list,
                                         n_radial=12,
                                         lmax=3,
                                         initial_sigma=1000.0,
                                         initial_sigma_e=1.0,
                                         initial_sigma_f=0.001,
                                         initial_sigma_s=1.0,
                                         variance_type=variance_type)

optimizer_configuration = FlareOptimizerConfiguration(optimization_method="nelder-mead",
                                                      max_optimization_iterations=10,
                                                      optimize_sigma=False,
                                                      optimize_sigma_e=False,
                                                      optimize_sigma_f=False,
                                                      optimize_sigma_s=False,
                                                      print=False)


if __name__ == '__main__':

    lammps_runner = LammpsRunner(lammps_executable_path=lammps_executable_path,
                                 mpi_processors=4,
                                 openmp_threads=4)

    artn_driver = ArtnDriver(lammps_runner=lammps_runner,
                             artn_library_plugin_path=artn_library_plugin_path,
                             reference_directory=reference_directory)

    oracle_calculator = StillingerWeberSinglePointCalculator(lammps_runner,
                                                             sw_coefficients_file_path)

    flare_optimizer = FlareHyperparametersOptimizer(optimizer_configuration)

    sample_maker_arguments = NoOpSampleMakerArguments(element_list=element_list)

    sample_maker = NoOpSampleMaker(sample_maker_arguments=sample_maker_arguments)

    active_learning = ActiveLearning(oracle_single_point_calculator=oracle_calculator,
                                     sample_maker=sample_maker,
                                     artn_driver=artn_driver,
                                     flare_hyperparameters_optimizer=flare_optimizer)

    flare_trainer = FlareTrainer(flare_configuration)

    # Add a single environment to the flare trainer
    # Add a random subset of environments from the 0th structure.
    initial_configuration_file_path = reference_directory / "initial_configuration.dat"
    initial_structure = LammpsData.from_file(str(initial_configuration_file_path),
                                             atom_style="atomic",
                                             sort_id=True).structure
    number_of_atoms = len(initial_structure)
    random_indices = np.random.randint(0, number_of_atoms, (8,))

    labelled_structure = oracle_calculator.calculate(initial_structure)

    active_environment_indices = list(random_indices)

    flare_trainer.add_labelled_structure(labelled_structure,
                                         active_environment_indices=active_environment_indices)

    # Run multiple campaigns with progressively more stringent uncertainty thresholds to train and refine
    # the FLARE model.
    for campaign_id, uncertainty_threshold in zip(list_campaign_ids, list_uncertainty_thresholds):
        active_learning_directory = experiment_dir / f"active_learning_campaign_{campaign_id}"

        active_learning.run_campaign(uncertainty_threshold=uncertainty_threshold,
                                     flare_trainer=flare_trainer,
                                     working_directory=active_learning_directory)

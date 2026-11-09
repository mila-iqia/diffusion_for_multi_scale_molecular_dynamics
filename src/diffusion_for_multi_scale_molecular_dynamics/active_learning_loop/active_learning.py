import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from ase import Atoms
from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.conversions import (
    convert_axl_to_structure_in_dict,
    convert_single_point_calculations_to_dataframe)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.logging import (
    CAMPAIGN_LOG_CONTEXT, clean_up_campaign_logger, set_up_campaign_logger)
from diffusion_for_multi_scale_molecular_dynamics.dynamic_driver.base_dynamic_driver import \
    DynamicDriver
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.outputs import (
    extract_all_fields_from_dump, extract_timesteps_from_dump)
from diffusion_for_multi_scale_molecular_dynamics.io.training_database import (
    Stage, TrainingDatabase)
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ORACLE_EVALUATION_PREFIX, UNCERTAIN_DUMP_FILENAME)
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import (  # noqa
    BaseSinglePointCalculator, SinglePointCalculation,
    get_active_environment_indices)
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.base_sample_maker import \
    BaseSampleMaker
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_conversion import (
    to_ase_atoms, to_pymatgen_structure)
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_converter import \
    StructureConverter

UNCERTAINTY_INFO_KEY = "uncertainty"
ACTIVE_INDICES_INFO_KEY = "active_indices"
CONSTRAINED_ATOM_INDICES_INFO_KEY = "constrained_atom_indices"


class ActiveLearning:
    """Active Learning.

    This class is the main driver of the active learning loop, dispatching sub-tasks as needed.

    Active learning flows as follows:
        - start with a MLIP that has been pretrained (ie, is not completely empty)
        - Iterate until SUCCESS:
            * deploy the MLIP
            * run the dynamic driver (ARTn or MD) with the MLIP:
                - SUCCESS if no encountered structure has an uncertainty above the threshold; exit.
                - INTERRUPTION otherwise (an uncertain structure was found).
            * collect the uncertain structure
            * generate structures to be evaluated by the Oracle
            * use the Oracle to evaluate the samples
            * add the labelled samples to the MLIP's training database
            * retrain the MLIP
    """

    def __init__(
        self,
        oracle_single_point_calculator: BaseSinglePointCalculator,
        sample_maker: BaseSampleMaker,
        dynamic_driver: DynamicDriver,
        mlip: BaseMLIP,
    ):
        """Init method.

        Args:
            oracle_single_point_calculator: class responsible for generating of ground truth labels.
            sample_maker: class responsible for generating samples for active learning.
            dynamic_driver: class responsible for running LAMMPS to search for uncertain structures (ARTn or MD).
            mlip: the machine-learning interatomic potential to drive and refine.
        """
        self.oracle_calculator = oracle_single_point_calculator
        self.sample_maker = sample_maker
        self.dynamic_driver = dynamic_driver
        self.mlip = mlip
        self._structure_converter = StructureConverter(list_of_element_symbols=sample_maker.arguments.element_list)

    def _get_uncertain_structure_and_uncertainties(
        self, dynamics_working_directory: Path, uncertainty_field: str
    ) -> Tuple[Atoms, np.ndarray, int]:
        """Get the uncertain configuration, its per-atom uncertainties and the step it was found at.

        This method assumes the CONVENTION that the dynamic driver's LAMMPS run produces a file
        named 'uncertain.dump' that contains the uncertain configuration.

        Args:
            dynamics_working_directory: directory holding the dynamic driver's 'uncertain.dump'.
            uncertainty_field: the per-atom uncertainty column to read (depends on the MLIP backend).
        """
        lammps_dump_path = dynamics_working_directory / UNCERTAIN_DUMP_FILENAME
        assert lammps_dump_path.is_file(), f"The file {lammps_dump_path} is missing."

        list_atoms, _, list_uncertainties = extract_all_fields_from_dump(
            lammps_dump_path, uncertainty_field=uncertainty_field
        )
        step = extract_timesteps_from_dump(lammps_dump_path)[0]
        return list_atoms[0], list_uncertainties[0], step

    def _make_samples(
        self, structure: Structure, uncertainty_per_atom: np.ndarray
    ) -> Tuple[List[Structure], List[np.array], List[Dict[str, Any]]]:
        """Make samples.

        This method handles the back-and-forth transformation from Pymatgen Structures to AXL structures.

        Ars:
            structure: Pymatgen structure to make samples from.
            uncertainty_per_atom: uncertainty per atom.

        Returns:
            list_sample_structures: list of sampled structures.
            list_active_indices: The indices of the active atoms in the sample structures.
            list_additional_information: list of additional information.
        """
        axl_structure = self._structure_converter.convert_structure_to_axl(structure)
        (list_sample_axl_structures,
         list_active_indices,
         list_sample_additional_information) = (
            self.sample_maker.make_samples(axl_structure, uncertainty_per_atom)
        )

        list_sample_structures = [
            self._structure_converter.convert_axl_to_structure(axl_structure)
            for axl_structure in list_sample_axl_structures
        ]
        converted_list_additional_information = [
            convert_axl_to_structure_in_dict(sample_info, self._structure_converter)
            for sample_info in list_sample_additional_information
        ]
        return (
            list_sample_structures,
            list_active_indices,
            converted_list_additional_information,
        )

    def _log_campaign_details(self, campaign_working_directory_path: Path, campaign_details: Dict):
        """Log campaign details."""
        output_file = campaign_working_directory_path / "campaign_details.yaml"
        with open(str(output_file), "w") as fd:
            yaml.dump(campaign_details, fd)

    def run_campaign(
        self,
        uncertainty_threshold: float,
        working_directory: Path,
        provided_configurations: List[Atoms],
        maximum_number_of_rounds: int = 100,
        restart_from_stage: str = "auto",
        initial_perturbation_standard_deviation: float = 0.05,
    ):
        """Run campaign.

        Perform a full campaign of active learning, resuming a crashed run when a database is already present.
        The preparation before the rounds (resume bookkeeping and precomputation) is done in
        ``_prepare_campaign``; each round is then run by ``_run_round``.

        Args:
            uncertainty_threshold: the uncertainty threshold to interrupt a dynamic driver run.
            working_directory: top directory where all the campaign artifacts are written.
            provided_configurations: the labelled starting configuration(s)
            maximum_number_of_rounds: maximum number of active learning rounds (guards against infinite loops).
            restart_from_stage: 'auto' resumes from what is on disk (or starts clean); 'driver'/'generate'/
                'oracle'/'train' force the resume stage of the latest epoch.
            initial_perturbation_standard_deviation: standard deviation (Angstrom) of the Gaussian
                displacements used to augment the seed during precomputation.
        """
        working_directory.mkdir(parents=True, exist_ok=True)
        logger = set_up_campaign_logger(working_directory)
        # To clean up the logger even if the run is interrupted, especially in a Jupyter notebook.
        try:
            logger.info("Starting Active Learning Simulation")
            self._logger = logger
            self._working_directory = working_directory
            self._uncertainty_threshold = uncertainty_threshold

            start_epoch, start_stage = self._prepare_campaign(
                restart_from_stage, provided_configurations, initial_perturbation_standard_deviation
            )

            epoch, stage = start_epoch, start_stage
            for _ in range(maximum_number_of_rounds):
                self._set_log_stage(epoch, stage)
                logger.info(f"Starting epoch {epoch} at stage {stage.name}")
                if self._run_round(epoch, stage):
                    logger.info("Active Learning Campaign is Complete. Exiting.")
                    break
                epoch += 1
                stage = Stage.DRIVER

            self._finish_campaign(uncertainty_threshold, epoch)
        finally:
            clean_up_campaign_logger(logger)

    def _prepare_campaign(
        self,
        restart_from_stage: str,
        provided_configurations: List[Atoms],
        standard_deviation: float,
    ) -> Tuple[int, Stage]:
        """Prepare the simulation state and return the (epoch, stage) to enter the round loop at.

        It restarts a simulation if precomputation or epoch_N is present. Else, it starts from scratch.
        """
        restart_epoch, restart_stage = TrainingDatabase.get_epoch_and_stage(
            self._working_directory, restart_from_stage
        )

        if restart_epoch == 0:
            self._logger.info("Starting from scratch.")
            return self._start_from_scratch(provided_configurations, standard_deviation)

        self._logger.info(f"Restarting from epoch {restart_epoch} at stage {restart_stage.name}.")
        return self._restart_computation(restart_epoch, restart_stage, restart_from_stage)

    def _start_from_scratch(
        self, provided_configurations: List[Atoms], standard_deviation: float
    ) -> Tuple[int, Stage]:
        """Start a fresh campaign: precompute the initial (epoch 0) model, then begin at the first round."""
        self._training_database = TrainingDatabase.from_scratch(self._working_directory)
        self._training_database.precomputation_model_directory()  # create the precomputation directory
        self.mlip.attach_training_database(self._training_database)
        self._run_precomputation(provided_configurations, standard_deviation)
        return 1, Stage.DRIVER

    def _restart_computation(
        self, restart_epoch: int, restart_stage: Stage, restart_from_stage: str
    ) -> Tuple[int, Stage]:
        """Restart from the work already on disk: reload the latest model and resume at the right stage."""
        self._training_database = TrainingDatabase.from_computation_folder(self._working_directory)
        self.mlip.attach_training_database(self._training_database)
        self._training_database.check_labelled_atoms_have_energy_and_forces()
        if restart_from_stage != "auto":
            self._training_database.reset_epoch_to_stage(restart_epoch, restart_stage)

        self.mlip.load(self._training_database.model_directory(restart_epoch - 1))  # epoch 0 = precomputation
        return restart_epoch, restart_stage

    def _finish_campaign(self, uncertainty_threshold: float, final_epoch: int) -> None:
        """Log the campaign summary once the round loop has finished."""
        campaign_details = dict(uncertainty_threshold=float(uncertainty_threshold),
                                final_epoch=int(final_epoch),
                                **self.mlip.training_metrics())
        self._log_campaign_details(campaign_working_directory_path=self._working_directory,
                                   campaign_details=campaign_details)

    def _run_precomputation(
        self, provided_configurations: List[Atoms], standard_deviation: float
    ) -> None:
        """Precompute the initial model before the first round.

        1. Let the MLIP seed its training database from the provided configuration(s).
        2. Fit and deploy the model. A model needing no training environments is simply deployed as-is.
        """
        self._set_log_stage(epoch=0, stage=Stage.TRAIN)  # precomputation is epoch 0, a training stage
        if not any(self.mlip.minimum_number_of_training_environments().values()):
            # Deploy the initial (empty) model into precomputation/ so restart's is_model_committed(0) finds it.
            self.mlip.prepare_mlip_first_round(self._training_database.precomputation_model_directory())
            return

        number_of_augmented = self.mlip.prepare_training_set(
            provided_configurations, self.oracle_calculator, standard_deviation
        )
        if number_of_augmented > 0:
            minimum_environments = self.mlip.minimum_number_of_training_environments()
            self._logger.info(
                f"Augmented the provided configurations with {number_of_augmented} perturbed, "
                f"{self.oracle_calculator.name}-labelled configurations to reach the active-set "
                f"minimum of {minimum_environments}."
            )

        self._logger.info("Precomputation: fitting and deploying the initial model.")
        model_directory = self._training_database.precomputation_model_directory()
        self._train_and_log(model_directory)
        self._update_latest_mlip_symlink(self._working_directory, model_directory)

    def _set_log_stage(self, epoch: int, stage: Stage) -> None:
        """Record the epoch/stage now running so every subsequent log line is prefixed with it."""
        # Pad the stage name so 'Train' lines up with the 6-character 'Driver'/'Oracle'.
        stage_label = stage.name.title().ljust(max(len(other.name) for other in Stage))
        CAMPAIGN_LOG_CONTEXT.set(epoch, stage_label)

    def _train_and_log(self, model_directory: Path) -> None:
        """Fit the MLIP, logging what it is trained on, how long the fit took and the resulting accuracy."""
        labelled_atoms = self._training_database.labelled_atoms
        number_of_configurations = len(labelled_atoms)
        number_of_environments = sum(len(atoms) for atoms in labelled_atoms)
        self._logger.info(
            f"Training {self.mlip.name} on {number_of_configurations} configurations "
            f"containing {number_of_environments} atomic environments."
        )
        training_program = self.mlip.training_program_name
        self._logger.info(f"Launching {training_program}.")
        start_time = time.time()
        self.mlip.train(model_directory)
        execution_time = time.time() - start_time
        self._logger.info(f"{training_program} training has finished. Execution Time: {execution_time:.3e} sec.")
        self._log_training_rmse()

    def _log_training_rmse(self) -> None:
        """Log the freshly trained model's training-set energy/forces RMSE (when available)."""
        metrics = self.mlip.training_metrics()
        if metrics["rmse_energy"] is not None and metrics["rmse_forces"] is not None:
            self._logger.info(
                f"RMSE(Energy) = {1000 * metrics['rmse_energy']:.1f} meV/at"
                f" RMSE(Force) = {1000 * metrics['rmse_forces']:.1f} meV/ang."
            )

    def _run_round(self, epoch: int, entry_stage: Stage) -> bool:
        """Run one round from entry_stage (skipping already-committed stages); return True when complete.

        A round is made of the following 6 steps, grouped into the 4 stages that also act as the restart
        indicators (a resume re-enters at a stage and reads the earlier stages' artifacts back from the
        database instead of recomputing them):
            Stage.DRIVER (run_dynamic_driver):
                1. Run the dynamic driver (ARTn or MD) with the MLIP.
                2. Extract the uncertainty per atom.
            Stage.GENERATE (generate_samples):
                3. Generate structures to be evaluated by the Oracle.
            Stage.ORACLE (oracle_evaluation):
                4. Evaluate the samples with the Oracle.
                5. Commit the labelled structures to the training database.
            Stage.TRAIN (_retrain):
                6. Fold the labelled structures into the model and retrain the MLIP.
        """
        current_stage = entry_stage
        if current_stage == Stage.DRIVER:
            self._set_log_stage(epoch, Stage.DRIVER)
            # Start this epoch's driver from a clean slate (clears any partial artifacts of a crashed attempt).
            self._training_database.reset_epoch_to_stage(epoch, Stage.DRIVER)
            uncertain_configuration = self.run_dynamic_driver(epoch)
            if uncertain_configuration is None:  # SUCCESS: no uncertain structure was found.
                return True
            self._training_database.write_dynamic(epoch, uncertain_configuration)
            current_stage = Stage.GENERATE

        if current_stage == Stage.GENERATE:
            self._set_log_stage(epoch, Stage.GENERATE)
            uncertain_configuration = self._training_database.read_dynamic(epoch)
            generated_samples = self.generate_samples(uncertain_configuration, epoch)
            generate_trajectory_path = self._training_database.write_generate(epoch, generated_samples)
            self._logger.info(f"Writing {len(generated_samples)} generated samples to {generate_trajectory_path}.")
            current_stage = Stage.ORACLE

        if current_stage == Stage.ORACLE:
            self._set_log_stage(epoch, Stage.ORACLE)
            generated_samples = self._training_database.read_generate(epoch)
            training_configurations = self.oracle_evaluation(generated_samples, epoch)
            oracle_trajectory_path = self._training_database.write_oracle(epoch, training_configurations)
            self._logger.info(f"Writing the labelled configurations to {oracle_trajectory_path}.")
            current_stage = Stage.TRAIN

        if current_stage == Stage.TRAIN:
            self._set_log_stage(epoch, Stage.TRAIN)
            training_configurations = self._training_database.read_oracle(epoch)
            self._retrain(epoch, training_configurations)

        return False

    def run_dynamic_driver(self, epoch: int) -> Optional[Atoms]:
        """Stage DRIVER (steps 1-2): run the driver and return the uncertain configuration (None on SUCCESS)."""
        dynamics_working_directory = self._training_database.dynamic_directory(epoch)

        self._logger.info("Launching the dynamic driver simulation...")
        calculation_state = self.dynamic_driver.run(
            mlip=self.mlip,
            working_directory=dynamics_working_directory,
            uncertainty_threshold=self._uncertainty_threshold,
        )
        self._logger.info(f"Dynamic driver state is {calculation_state}")

        if calculation_state == CalculationState.ERROR:
            raise RuntimeError(
                f"The dynamic driver run failed (state ERROR). Review the logs in {dynamics_working_directory}."
            )
        if calculation_state == CalculationState.SUCCESS:
            for summary_line in self.dynamic_driver.summarize_success(dynamics_working_directory):
                self._logger.info(summary_line)
            return None

        uncertain_configuration, uncertainty_per_atom, step = self._get_uncertain_structure_and_uncertainties(
            dynamics_working_directory, self.mlip.lammps_potential.uncertainty_field()
        )
        for summary_line in self.dynamic_driver.summarize_interruption(dynamics_working_directory):
            self._logger.info(summary_line)
        number_of_flagged_environments = int(np.sum(uncertainty_per_atom > self._uncertainty_threshold))
        self._logger.info(self._flagged_environments_message(step, number_of_flagged_environments))

        uncertain_configuration.info[UNCERTAINTY_INFO_KEY] = np.asarray(uncertainty_per_atom, dtype=float)
        return uncertain_configuration

    def _flagged_environments_message(self, step: int, number_of_flagged_environments: int) -> str:
        """Report the step (X/Y when the driver has a step budget, X otherwise) and how many were flagged."""
        maximum_number_of_steps = self.dynamic_driver.maximum_number_of_steps
        step_label = (
            f"Lammps step {step}" if maximum_number_of_steps is None
            else f"Lammps step {step}/{maximum_number_of_steps}"
        )
        return f"{step_label} flagged {number_of_flagged_environments} atomic environments above the threshold."

    def generate_samples(self, uncertain_configuration: Atoms, epoch: int) -> List[Atoms]:
        """Stage GENERATE (step 3): generate structures to be evaluated by the Oracle."""
        uncertain_structure = to_pymatgen_structure(uncertain_configuration)
        uncertainty_per_atom = uncertain_configuration.info[UNCERTAINTY_INFO_KEY]

        list_sample_structures, list_active_indices, list_sample_information = self._make_samples(
            uncertain_structure, uncertainty_per_atom
        )

        list_generated_samples = []
        for structure, active_indices, sample_information in zip(
            list_sample_structures, list_active_indices, list_sample_information
        ):
            atoms = to_ase_atoms(structure)
            atoms.info[ACTIVE_INDICES_INFO_KEY] = np.asarray(active_indices)
            atoms.info[CONSTRAINED_ATOM_INDICES_INFO_KEY] = np.asarray(
                sample_information["constrained_atom_indices"]
            )
            list_generated_samples.append(atoms)
        return list_generated_samples

    def oracle_evaluation(self, generated_samples: List[Atoms], epoch: int) -> List[Atoms]:
        """Stage ORACLE (steps 4-5): evaluate the generated samples with the Oracle and label them."""
        list_active_indices = [atoms.info[ACTIVE_INDICES_INFO_KEY] for atoms in generated_samples]
        list_sample_information = [
            {"constrained_atom_indices": atoms.info[CONSTRAINED_ATOM_INDICES_INFO_KEY]}
            for atoms in generated_samples
        ]

        self._logger.info(
            f"Labelling {len(generated_samples)} new configurations with {self.oracle_calculator.name}."
        )
        oracle_directory = self._training_database.oracle_directory(epoch)
        start_time = time.time()
        list_single_point_calculations = []
        for index, atoms in enumerate(generated_samples):
            results_path = oracle_directory / f"{ORACLE_EVALUATION_PREFIX}_{index}.dump"
            evaluation_start_time = time.time()
            calculation = self.oracle_calculator.calculate(atoms, results_path=results_path)
            list_single_point_calculations.append(calculation)
            self._logger.info(
                f"{self.oracle_calculator.name} evaluation {index + 1}/{len(generated_samples)} completed in "
                f"{time.time() - evaluation_start_time:.1f} s with energy = {calculation.energy:.4f} eV."
            )
        self._logger.info(f"Labelling has finished. Execution Time: {time.time() - start_time:.3e} sec.")

        oracle_dataframe = convert_single_point_calculations_to_dataframe(
            list_single_point_calculations, list_sample_information
        )
        oracle_dataframe.to_pickle(oracle_directory / "oracle_single_point_calculations.pkl")

        return [
            calculation.to_atoms(active_environment_indices)
            for calculation, active_environment_indices in zip(list_single_point_calculations, list_active_indices)
        ]

    def _retrain(self, epoch: int, training_configurations: List[Atoms]) -> None:
        """Stage TRAIN (step 6): fold this epoch's labelled data into the model, retrain, commit the model."""
        for atoms in training_configurations:
            self.mlip.add_labelled_structure(
                SinglePointCalculation.from_atoms(atoms), get_active_environment_indices(atoms)
            )
        model_directory = self._training_database.model_directory(epoch)
        self._train_and_log(model_directory)
        self._update_latest_mlip_symlink(self._working_directory, model_directory)

    @staticmethod
    def _update_latest_mlip_symlink(working_directory: Path, mlip_training_directory: Path):
        """Point 'latest_mlip' at the most recent MLIP training directory."""
        symlink_path = working_directory / "latest_mlip"
        if symlink_path.is_symlink() or symlink_path.exists():
            symlink_path.unlink()
        symlink_path.symlink_to(mlip_training_directory.resolve(), target_is_directory=True)

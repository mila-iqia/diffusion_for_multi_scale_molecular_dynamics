import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.artn.calculation_state import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.dynamic_driver.artn_driver import \
    ArtnDriver
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.outputs import \
    extract_all_fields_from_dump
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.logging import (
    clean_up_campaign_logger, set_up_campaign_logger)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.base_sample_maker import \
    BaseSampleMaker
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.namespace import (
    AXL_STRUCTURE_IN_NEW_BOX, AXL_STRUCTURE_IN_ORIGINAL_BOX)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.structure_converter import \
    StructureConverter
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_single_point_calculator import (  # noqa
    BaseSinglePointCalculator, SinglePointCalculation)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_hyperparameter_optimizer import \
    FlareHyperparametersOptimizer
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import \
    FlareTrainer


class ActiveLearning:
    """Active Learning.

    This class is the main driver of the active learning loop, dispatching sub-tasks as needed.

    Active learning flows as follows:
        - start with a FLARE sparse Gaussian Process (SGP) model that has been pretrained (ie, is not completely empty)
        - Iterate until SUCCESS:
            * map SGP
            * drive artn with mapped SGP; if SUCCESS -> exit.
            * collect uncertain structure
            * make samples from uncertain structure
            * label samples
            * add labels to SGP; retrain SGP
    """

    def __init__(
        self,
        oracle_single_point_calculator: BaseSinglePointCalculator,
        sample_maker: BaseSampleMaker,
        artn_driver: ArtnDriver,
        flare_hyperparameters_optimizer: FlareHyperparametersOptimizer
    ):
        """Init method.

        Args:
            oracle_single_point_calculator: class responsible for generating of ground truth labels.
            sample_maker: class responsible for generating samples for active learning.
            artn_driver: class responsible for running LAMMPS + ARTn.
            flare_hyperparameters_optimizer: class responsible for learning the model's hyperparameters.

        """
        self.oracle_calculator = oracle_single_point_calculator
        self.sample_maker = sample_maker
        self.artn_driver = artn_driver
        self.optimizer = flare_hyperparameters_optimizer
        self._structure_converter = StructureConverter(list_of_element_symbols=sample_maker.arguments.element_list)

    def _get_uncertain_structure_and_uncertainties(
        self, artn_working_directory: Path
    ) -> Tuple[Structure, np.ndarray]:
        """Get uncertain structure.

        This method assumes the CONVENTION that the ARTn + LAMMPS run will produce a file
        named 'uncertain_dump.yaml' that contains the uncertain structure.
        """
        lammps_dump_path = artn_working_directory / "uncertain_dump.yaml"
        assert lammps_dump_path.is_file(), f"The file {lammps_dump_path} is missing."

        list_structures, _, _, list_uncertainties = extract_all_fields_from_dump(
            lammps_dump_path
        )
        uncertain_structure = list_structures[0]
        uncertainties = list_uncertainties[0]
        return uncertain_structure, uncertainties

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
            self._convert_axl_to_structure_in_dict(sample_info)
            for sample_info in list_sample_additional_information
        ]
        return (
            list_sample_structures,
            list_active_indices,
            converted_list_additional_information,
        )

    def _convert_axl_to_structure_in_dict(
        self, sample_additional_information: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert AXL elements of an additional information dictionary to pymatgen structure.

        Args:
            sample_additional_information: additional information about a sample in a dictionary.

        Returns:
            new_structures: additional information about a sample in a dictionary with AXL as Structure
        """
        converted_info = {}
        for k, v in sample_additional_information.items():
            if k in [AXL_STRUCTURE_IN_ORIGINAL_BOX, AXL_STRUCTURE_IN_NEW_BOX]:
                converted_info[k] = self._structure_converter.convert_axl_to_structure(v)
            else:
                converted_info[k] = v
        return converted_info

    def _convert_single_point_calculations_to_dataframe(
        self,
        list_single_point_calculations: List[SinglePointCalculation],
        list_sample_information: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Convert single point calculations to dataframe."""
        rows = []
        for calculation, sample_information in zip(
            list_single_point_calculations, list_sample_information
        ):
            row = dict(
                calculation_type=calculation.calculation_type,
                structure=calculation.structure,
                forces=calculation.forces,
                energy=calculation.energy,
                sample_information=sample_information,
            )
            rows.append(row)

        df = pd.DataFrame(data=rows)
        return df

    def _log_campaign_details(self, campaign_working_directory_path: Path, campaign_details: Dict):
        """Log campaign details."""
        output_file = campaign_working_directory_path / "campaign_details.yaml"
        with open(str(output_file), "w") as fd:
            yaml.dump(campaign_details, fd)

    def run_campaign(
        self,
        uncertainty_threshold: float,
        flare_trainer: FlareTrainer,
        working_directory: Path,
        maximum_number_of_rounds: int = 100,
    ):
        """Run campaign.

        Perform a full campaign of active learning.

        Args:
            uncertainty_threshold: the uncertainty threshold to interrupt an ARTn run.
            flare_trainer: the class containing the sparse Gaussian Process (SGP). It is assumed that this model
                is already somewhat pretrained (ie, at least one structure) so that it can be invoked. In other
                words, this is not a completely empty SGP.
            working_directory: top directory where all the various artifacts from this campaign will be written.
            maximum_number_of_rounds: maximum number of active learning rounds. This is useful to avoid
                infinite loops...
        """
        working_directory.mkdir(parents=True, exist_ok=True)
        logger = set_up_campaign_logger(working_directory)
        logger.info("Starting Active Learning Simulation")

        round_number = 0

        while round_number <= maximum_number_of_rounds:
            round_number += 1
            logger.info(f"Starting Round {round_number}")

            current_sub_directory = working_directory / f"round_{round_number}"

            mapped_coefficients_directory = (
                current_sub_directory / "FLARE_mapped_coefficients"
            )
            mapped_coefficients_directory.mkdir(parents=True, exist_ok=True)

            # The artn_driver will create this directory.
            artn_working_directory = current_sub_directory / "lammps_artn"

            pair_coeff_file_path, mapped_uncertainty_file_path = (
                flare_trainer.write_mapped_model_to_disk(
                    mapped_coefficients_directory, version=round_number
                )
            )

            logger.info("  Launching ARTn simulation...")
            calculation_state = self.artn_driver.run(
                working_directory=artn_working_directory,
                uncertainty_threshold=uncertainty_threshold,
                pair_coeff_file_path=pair_coeff_file_path,
                mapped_uncertainty_file_path=mapped_uncertainty_file_path,
            )
            logger.info(f"  ARTn state is {calculation_state}")

            if calculation_state == CalculationState.SUCCESS:
                logger.info("Active Learning Campaign is Complete.")

                logger.info("Writing FLARE model checkpoint.")
                checkpoint_path = working_directory / "trained_flare.json"
                flare_trainer.write_checkpoint_to_disk(checkpoint_path)
                logger.info("Exiting.")
                break

            logger.info("  Extracting uncertain structure from ARTn work directory...")
            uncertain_structure, uncertainty_per_atom = (
                self._get_uncertain_structure_and_uncertainties(artn_working_directory)
            )

            number_of_uncertain_envs = np.sum(
                uncertainty_per_atom > uncertainty_threshold
            )
            logger.info(
                f" -> There are {number_of_uncertain_envs} environments with uncertainty above the threshold."
            )

            logger.info("  Making new samples based on uncertainties.")
            list_sample_structures, list_active_indices, list_sample_information = (
                self._make_samples(uncertain_structure, uncertainty_per_atom))

            logger.info("  Labelling samples with oracle...")
            time1 = time.time()
            list_single_point_calculations = [
                self.oracle_calculator.calculate(structure) for structure in list_sample_structures]
            time2 = time.time()
            logger.info(
                f" -> It took {time2- time1: 6.2e} seconds to compute labels with Oracle."
            )

            logger.info("  Converting labelled samples and writing pickle to disk.")
            oracle_df = self._convert_single_point_calculations_to_dataframe(
                list_single_point_calculations, list_sample_information
            )

            oracle_directory = current_sub_directory / "oracle"
            oracle_directory.mkdir(parents=True, exist_ok=True)

            output_file = oracle_directory / "oracle_single_point_calculations.pkl"
            oracle_df.to_pickle(output_file)

            logger.info("  Adding samples and uncertain environment to FLARE.")
            for single_point_calculation, active_environment_indices \
                    in zip(list_single_point_calculations, list_active_indices):
                flare_trainer.add_labelled_structure(
                    single_point_calculation,
                    active_environment_indices=active_environment_indices,
                )

            if self.optimizer.is_inactive:
                logger.info("  The optimizer is inactive: no hyperparameter training is done.")

            else:
                logger.info("  Fitting the FLARE hyperparameters...")
                optimization_result, history_df = flare_trainer.fit_hyperparameters(self.optimizer)
                logger.info(f"  Optimization status : {optimization_result.success}")
                logger.info(f"  Optimization message : {optimization_result.message}")
                hyperparameter_optimization_log = current_sub_directory / "hyperparameter_optimization_logs"
                hyperparameter_optimization_log.mkdir(parents=True, exist_ok=True)
                history_df.to_pickle(hyperparameter_optimization_log / "optimization_log.pkl")

            # TODO: this logging could be encapsulated better in a FLARE object.
            logger.info("  The SGP hyperparameters are now : ")
            sigma, sigma_e, sigma_f, sigma_s = flare_trainer.sgp_model.sparse_gp.hyperparameters
            logger.info(f"       sigma   = {sigma: 12.8f}")
            logger.info(f"       sigma_e = {sigma_e: 12.8f}")
            logger.info(f"       sigma_f = {sigma_f: 12.8f}")
            logger.info(f"       sigma_s = {sigma_s: 12.8f}")

        sigma, sigma_e, sigma_f, sigma_s = flare_trainer.sgp_model.sparse_gp.hyperparameters
        campaign_details = dict(uncertainty_threshold=float(uncertainty_threshold),
                                final_round=int(round_number),
                                sigma=float(sigma),
                                sigma_e=float(sigma_e),
                                sigma_f=float(sigma_f),
                                sigma_s=float(sigma_s))

        self._log_campaign_details(campaign_working_directory_path=working_directory,
                                   campaign_details=campaign_details)
        # Delete the logger to avoid overlogging across campaigns.
        clean_up_campaign_logger(logger)

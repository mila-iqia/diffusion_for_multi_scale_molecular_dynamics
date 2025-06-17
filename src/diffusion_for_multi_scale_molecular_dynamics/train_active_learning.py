"""Entry point to apply Active Learning."""

import argparse
import logging
import time
import typing
from pathlib import Path

import lightning as pl

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.active_learning import \
    ActiveLearning
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.dynamic_driver.artn_driver import \
    ArtnDriver
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.excisor_factory import \
    create_excisor_parameters
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.lammps_runner import \
    instantiate_lammps_runner
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.base_sample_maker import \
    BaseSampleMaker
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.sample_maker_factory import (
    create_sample_maker, create_sample_maker_parameters)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.single_point_calculator_factory import \
    instantiate_single_point_calculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_hyperparameter_optimizer import (
    FlareHyperparametersOptimizer, FlareOptimizerConfiguration)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import \
    FlareTrainer
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    configure_logging
from diffusion_for_multi_scale_molecular_dynamics.utils.main_utils import \
    load_and_backup_hyperparameters

logger = logging.getLogger(__name__)


def main(args: typing.Optional[typing.Any] = None):
    """Create an active learning experiment : main entry point of the program."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        help="path to configuration file with parameters defining the task in yaml format",
        required=True,
    )

    parser.add_argument(
        "--path_to_reference_directory",
        help="path to a directory that contains the ART input file and initial configuration. "
        "This defines the task to be accomplished.",
        default=None,
        required=True,
    )

    parser.add_argument(
        "--path_to_lammps_executable",
        help="path to a LAMMPS executable that is compatible with ARTn and FLARE.",
        default=None,
        required=True,
    )

    parser.add_argument(
        "--path_to_artn_library_plugin",
        help="path to the compiled ARTn_plugin library.",
        default=None,
        required=True,
    )

    parser.add_argument(
        "--path_to_initial_flare_checkpoint",
        help="path to a FLARE model checkpoint that has been pretrained (ie, is not empty).",
        default=None,
        required=True,
    )

    parser.add_argument(
        "--output_directory",
        help="path to where the outputs will be written.",
        required=True,
    )
    args = parser.parse_args(args)

    output_directory = Path(args.output_directory)

    if output_directory.is_dir():
        raise Exception(
            f"Output directory {args.output_directory} already exists! Stopping to avoid overwriting data."
        )
    output_directory.mkdir(parents=True, exist_ok=False)

    configuration = load_and_backup_hyperparameters(
        config_file_path=args.config, output_directory=args.output_directory
    )

    run(args, configuration)


def run(args: argparse.Namespace, configuration: typing.Dict):
    """Create and run the active learning experiment.

    Args:
        args (object): arguments passed from the cli
        configuration (dict): parameters from the config file
    """
    configure_logging(experiment_dir=args.output_directory,
                      logger=logger,
                      log_to_console=True)

    experiment_name = configuration["exp_name"]
    logging.info(f"Starting experiment {experiment_name}")

    if "seed" in configuration:
        pl.seed_everything(configuration["seed"])

    element_list = configuration["elements"]
    ElementTypes.validate_elements(element_list)

    lammps_runner = instantiate_lammps_runner(lammps_executable_path=Path(args.path_to_lammps_executable),
                                              configuration_dict=configuration)

    artn_driver = ArtnDriver(
        lammps_runner=lammps_runner,
        artn_library_plugin_path=Path(args.path_to_artn_library_plugin),
        reference_directory=Path(args.path_to_reference_directory).absolute(),
    )

    assert (
        "oracle" in configuration
    ), "An Oracle must be defined in the configuration file!"
    oracle_configuration = configuration["oracle"]
    oracle_calculator = instantiate_single_point_calculator(
        single_point_calculator_configuration=oracle_configuration,
        lammps_runner=lammps_runner)

    assert (
        "flare" in configuration
    ), "An Flare configuration must be defined in the configuration file!"
    flare_parameters = configuration["flare"]
    optimizer_parameters = flare_parameters.pop("flare_optimizer")
    optimize_on_the_fly = optimizer_parameters.pop("optimize_on_the_fly")

    if optimize_on_the_fly:
        flare_optimizer_configuration = FlareOptimizerConfiguration(
            **optimizer_parameters
        )
    else:
        flare_optimizer_configuration = FlareOptimizerConfiguration(
            optimize_sigma=False,
            optimize_sigma_e=False,
            optimize_sigma_f=False,
            optimize_sigma_s=False,
        )

    flare_optimizer = FlareHyperparametersOptimizer(flare_optimizer_configuration)

    assert "sampling" in configuration, "A sampling strategy for must be defined in the configuration file!"
    sampling_dictionary = configuration["sampling"]
    sample_maker = get_sample_maker_from_configuration(sampling_dictionary, element_list)

    active_learning = ActiveLearning(
        oracle_single_point_calculator=oracle_calculator,
        sample_maker=sample_maker,
        artn_driver=artn_driver,
        flare_hyperparameters_optimizer=flare_optimizer,
    )

    assert (
        "uncertainty_thresholds" in configuration
    ), "A list of uncertainty thresholds must be defined in the configuration file!"
    uncertainty_thresholds = configuration["uncertainty_thresholds"]

    list_flare_checkpoint_paths = [Path(args.path_to_initial_flare_checkpoint).absolute()]

    try:
        for campaign_id, uncertainty_threshold in enumerate(uncertainty_thresholds, 1):
            logger.info(f"Starting campaign {campaign_id} uncertainty threshold {uncertainty_threshold}")
            checkpoint_path = list_flare_checkpoint_paths[-1]
            logger.info(f"  - Loading checkpoint from {checkpoint_path}")
            flare_trainer = FlareTrainer.from_checkpoint(checkpoint_path)

            working_directory = Path(args.output_directory).absolute() / f"campaign_{campaign_id}"
            working_directory.mkdir(parents=True, exist_ok=False)
            time1 = time.time()
            active_learning.run_campaign(
                uncertainty_threshold=uncertainty_threshold,
                flare_trainer=flare_trainer,
                working_directory=working_directory,
            )
            time2 = time.time()
            logger.info(f"Campaign {campaign_id} completed in {time2-time1: 6.2f} seconds.")

            new_checkpoint_path = working_directory / "trained_flare.json"
            assert new_checkpoint_path.is_file(), \
                f"The checkpoint file at the end of campaign {campaign_id} is missing! Something went wrong."
            list_flare_checkpoint_paths.append(new_checkpoint_path)

    except RuntimeError as err:
        logger.error(err)


def get_sample_maker_from_configuration(sampling_dictionary: typing.Dict,
                                        element_list: typing.List[str]) -> BaseSampleMaker:
    """Get sample maker from configuration dictionary."""
    # TODO: deal with a potential diffusion model.
    excisor_parameter_dictionary = sampling_dictionary.pop("excision", None)
    excisor_parameters = create_excisor_parameters(excisor_parameter_dictionary)
    sampling_dictionary["element_list"] = element_list
    sample_maker_parameters = create_sample_maker_parameters(sampling_dictionary)
    sample_maker = create_sample_maker(sample_maker_parameters=sample_maker_parameters,
                                       excisor_parameters=excisor_parameters)
    return sample_maker


if __name__ == "__main__":
    # Uncomment the following in order to use Pycharm's Remote Debugging server, which allows to
    # launch python commands through a bash script (and through Orion!). VERY useful for debugging.
    # This requires a professional edition of Pycharm and installing the pydevd_pycharm package with pip.
    # The debug server stopped working in 2024.3. There is a workaround. See:
    #   https://www.reddit.com/r/pycharm/comments/1gs1lgk/python_debug_server_issues/
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=56636, stdoutToServer=True, stderrToServer=True)
    main()

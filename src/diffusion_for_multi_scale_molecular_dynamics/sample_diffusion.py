"""Sample Diffusion.

This script is the entry point to draw samples from a pre-trained model checkpoint.
"""

import argparse
import logging
import os
import socket
from pathlib import Path
from typing import Any, AnyStr, Dict, Optional, Union

import torch

from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import \
    SamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.constrained_langevin_generator import \
    ConstrainedPredictorCorrectorAXLGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.instantiate_generator import \
    instantiate_generator
from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.load_sampling_parameters import \
    load_sampling_parameters
from diffusion_for_multi_scale_molecular_dynamics.models.axl_diffusion_lightning_model import \
    AXLDiffusionLightningModel
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.oracle.energy_oracle import \
    OracleParameters
from diffusion_for_multi_scale_molecular_dynamics.oracle.energy_oracle_factory import (
    create_energy_oracle, create_energy_oracle_parameters)
from diffusion_for_multi_scale_molecular_dynamics.sampling.diffusion_sampling import \
    create_batch_of_samples
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import (
    get_git_hash, setup_console_logger)
from diffusion_for_multi_scale_molecular_dynamics.utils.main_utils import \
    load_and_backup_hyperparameters

logger = logging.getLogger(__name__)


def main(args: Optional[Any] = None):
    """Load a diffusion model and draw samples.

    This main.py file is meant to be called using the cli.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="config file with sampling parameters in yaml format.",
    )
    parser.add_argument(
        "--path_to_constraint_data_pickle", required=False,
        help="path to a pickle that contains a reference compositions and fixed atom indices."
    )

    parser.add_argument(
        "--checkpoint", required=True, help="path to checkpoint model to be loaded."
    )
    parser.add_argument(
        "--output", required=True, help="path to outputs - will store files here"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to use. Defaults to cuda."
    )
    args = parser.parse_args(args)
    if os.path.exists(args.output):
        logger.info(f"WARNING: the output directory {args.output} already exists!")
    else:
        os.makedirs(args.output)

    setup_console_logger(experiment_dir=args.output)
    assert os.path.exists(
        args.checkpoint
    ), f"The path {args.checkpoint} does not exist. Cannot go on."

    script_location = os.path.realpath(__file__)
    git_hash = get_git_hash(script_location)
    hostname = socket.gethostname()
    logger.info("Sampling Experiment info:")
    logger.info(f"  Hostname : {hostname}")
    logger.info(f"  Git Hash : {git_hash}")
    logger.info(f"  Checkpoint : {args.checkpoint}")

    # Very opinionated logger, which writes to the output folder.
    logger.info(f"Start Generating Samples with checkpoint {args.checkpoint}")

    hyper_params = load_and_backup_hyperparameters(
        config_file_path=args.config, output_directory=args.output
    )

    device = torch.device(args.device)
    noise_parameters, sampling_parameters = extract_and_validate_parameters(
        hyper_params
    )

    if "elements" in hyper_params:
        ElementTypes.validate_elements(hyper_params["elements"])

    oracle_parameters = None
    if "oracle" in hyper_params:
        assert "elements" in hyper_params, \
            "elements are needed to define the energy oracle."
        elements = hyper_params["elements"]
        oracle_parameters = create_energy_oracle_parameters(hyper_params["oracle"], elements)

    axl_network = get_axl_network(args.checkpoint)

    logger.info("Instantiate generator...")
    raw_generator = instantiate_generator(
        sampling_parameters=sampling_parameters,
        noise_parameters=noise_parameters,
        axl_network=axl_network,
    )

    if args.path_to_constraint_data_pickle:
        logger.info("Constrained Sampling is activated")
        constraint_data_pickle_path = Path(args.path_to_constraint_data_pickle)
        assert constraint_data_pickle_path.is_file(), "The constraint data pickle does not exist."

        constraint_data = torch.load(constraint_data_pickle_path)
        constrained_atom_indices = constraint_data["constrained_atom_indices"]
        logger.info(f"Constrained atom indices are {constrained_atom_indices}")

        reference_composition = constraint_data["reference_composition"]

        generator = ConstrainedPredictorCorrectorAXLGenerator(raw_generator,
                                                              reference_composition,
                                                              constrained_atom_indices)
    else:
        generator = raw_generator

    create_samples_and_write_to_disk(
        generator=generator,
        sampling_parameters=sampling_parameters,
        oracle_parameters=oracle_parameters,
        device=device,
        output_path=args.output,
    )


def extract_and_validate_parameters(hyper_params: Dict[AnyStr, Any]):
    """Extract and validate parameters.

    Args:
        hyper_params : Dictionary of hyper-parameters for drawing samples.

    Returns:
        noise_parameters: object that defines the noise schedule
        sampling_parameters: object that defines how to draw samples, and how many.
    """
    assert (
        "noise" in hyper_params
    ), "The noise parameters must be defined to draw samples."
    noise_parameters = NoiseParameters(**hyper_params["noise"])

    assert (
        "sampling" in hyper_params
    ), "The sampling parameters must be defined to draw samples."
    sampling_parameters = load_sampling_parameters(hyper_params["sampling"])

    return noise_parameters, sampling_parameters


def get_axl_network(checkpoint_path: Union[str, Path]) -> ScoreNetwork:
    """Get AXL network.

    Args:
        checkpoint_path : path where the checkpoint is written.

    Returns:
        axl network network: read from the checkpoint.
    """
    logger.info("Loading checkpoint...")
    pl_model = AXLDiffusionLightningModel.load_from_checkpoint(checkpoint_path)
    pl_model.eval()

    axl_network = pl_model.axl_network
    return axl_network


def create_samples_and_write_to_disk(
    generator: LangevinGenerator,
    sampling_parameters: SamplingParameters,
    oracle_parameters: Union[OracleParameters, None],
    device: torch.device,
    output_path: Union[str, Path],
):
    """Create Samples and write to disk.

    Method that drives the creation of samples.

    Args:
        noise_parameters: object that defines the noise schedule
        sampling_parameters: object that defines how to draw samples, and how many.
        device: which device should be used to draw samples.
        checkpoint_path : path to checkpoint of model to be loaded.
        output_path: where the outputs should be written.

    Returns:
        None
    """
    logger.info("Generating samples...")
    with torch.no_grad():
        samples_batch = create_batch_of_samples(
            generator=generator,
            sampling_parameters=sampling_parameters,
            device=device,
        )
    logger.info("Done Generating Samples.")

    logger.info("Writing samples to disk...")
    output_directory = Path(output_path)
    with open(output_directory / "samples.pt", "wb") as fd:
        torch.save(samples_batch, fd)

    if oracle_parameters:
        logger.info("Compute energy from Oracle...")
        oracle = create_energy_oracle(oracle_parameters)
        sample_energies = oracle.compute_oracle_energies(samples_batch)

        logger.info("Writing energies to disk...")
        with open(output_directory / "energies.pt", "wb") as fd:
            torch.save(sample_energies, fd)

    if sampling_parameters.record_samples:
        logger.info("Writing sampling trajectories to disk...")
        generator.sample_trajectory_recorder.write_to_pickle(
            output_directory / "trajectories.pt"
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()

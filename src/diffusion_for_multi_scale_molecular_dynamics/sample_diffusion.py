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

from diffusion_for_multi_scale_molecular_dynamics.generators.instantiate_generator import \
    instantiate_generator
from diffusion_for_multi_scale_molecular_dynamics.generators.load_sampling_parameters import \
    load_sampling_parameters
from diffusion_for_multi_scale_molecular_dynamics.generators.position_generator import \
    SamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.main_utils import \
    load_and_backup_hyperparameters
from diffusion_for_multi_scale_molecular_dynamics.models.position_diffusion_lightning_model import \
    PositionDiffusionLightningModel
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.oracle.energies import \
    compute_oracle_energies
from diffusion_for_multi_scale_molecular_dynamics.samplers.variance_sampler import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.samples.sampling import \
    create_batch_of_samples
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import (
    get_git_hash, setup_console_logger)

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

    create_samples_and_write_to_disk(
        noise_parameters=noise_parameters,
        sampling_parameters=sampling_parameters,
        device=device,
        checkpoint_path=args.checkpoint,
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


def get_sigma_normalized_score_network(
    checkpoint_path: Union[str, Path]
) -> ScoreNetwork:
    """Get sigma-normalized score network.

    Args:
        checkpoint_path : path where the checkpoint is written.

    Returns:
        sigma_normalized score network: read from the checkpoint.
    """
    logger.info("Loading checkpoint...")
    pl_model = PositionDiffusionLightningModel.load_from_checkpoint(checkpoint_path)
    pl_model.eval()

    sigma_normalized_score_network = pl_model.sigma_normalized_score_network
    return sigma_normalized_score_network


def create_samples_and_write_to_disk(
    noise_parameters: NoiseParameters,
    sampling_parameters: SamplingParameters,
    device: torch.device,
    checkpoint_path: Union[str, Path],
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
    sigma_normalized_score_network = get_sigma_normalized_score_network(checkpoint_path)

    logger.info("Instantiate generator...")
    position_generator = instantiate_generator(
        sampling_parameters=sampling_parameters,
        noise_parameters=noise_parameters,
        sigma_normalized_score_network=sigma_normalized_score_network,
    )

    logger.info("Generating samples...")
    with torch.no_grad():
        samples_batch = create_batch_of_samples(
            generator=position_generator,
            sampling_parameters=sampling_parameters,
            device=device,
        )
    logger.info("Done Generating Samples.")

    logger.info("Writing samples to disk...")
    output_directory = Path(output_path)
    with open(output_directory / "samples.pt", "wb") as fd:
        torch.save(samples_batch, fd)

    logger.info("Compute energy from Oracle...")
    sample_energies = compute_oracle_energies(samples_batch)

    logger.info("Writing energies to disk...")
    with open(output_directory / "energies.pt", "wb") as fd:
        torch.save(sample_energies, fd)

    if sampling_parameters.record_samples:
        logger.info("Writing sampling trajectories to disk...")
        position_generator.sample_trajectory_recorder.write_to_pickle(
            output_directory / "trajectories.pt"
        )


if __name__ == "__main__":
    main()
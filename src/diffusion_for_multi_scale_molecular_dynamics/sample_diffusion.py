"""Sample Diffusion.

This script is the entry point to draw samples from a pre-trained model checkpoint.
"""

import argparse
import logging
import os
import socket
from pathlib import Path
from typing import Any, AnyStr, Dict, Optional, Union

import orion.client
import torch

from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import \
    SamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.instantiate_generator import \
    instantiate_generator
from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.load_sampling_parameters import \
    load_sampling_parameters
from diffusion_for_multi_scale_molecular_dynamics.generators.trajectory_initializer import \
    instantiate_trajectory_initializer
from diffusion_for_multi_scale_molecular_dynamics.models.axl_diffusion_lightning_model import \
    AXLDiffusionLightningModel
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.force_field_augmented_score_network import (
    ForceFieldAugmentedScoreNetwork, ForceFieldParameters)
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


def main(args: Optional[Any] = None, axl_network: Optional[ScoreNetwork] = None) -> None:
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
        "--checkpoint", default=None, help="path to checkpoint model to be loaded."
    )
    parser.add_argument(
        "--output", required=True, help="path to outputs - will store files here"
    )

    parser.add_argument(
        "--path_to_constraint_data_pickle", default=None,
        help="Path to a pickle that contains reference compositions and starting time index."
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

    if axl_network is None:
        assert os.path.exists(
            args.checkpoint
        ), f"The path {args.checkpoint} does not exist. Cannot go on."

    script_location = os.path.realpath(__file__)
    git_hash = get_git_hash(script_location)
    hostname = socket.gethostname()
    device = torch.device(args.device)
    logger.info("Sampling Experiment info:")
    logger.info(f"  Hostname : {hostname}")
    logger.info(f"  Git Hash : {git_hash}")
    logger.info(f"  Checkpoint : {args.checkpoint}")
    logger.info(f"  Device   : {device}")
    if args.path_to_constraint_data_pickle:
        logger.info(f"  Constraint Pickle : {args.path_to_constraint_data_pickle}")

    hyper_params = load_and_backup_hyperparameters(
        config_file_path=args.config, output_directory=args.output
    )

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

    if axl_network is None:
        # Very opinionated logger, which writes to the output folder.
        logger.info(f"Start Generating Samples with checkpoint {args.checkpoint}")
        axl_network = get_axl_network(args.checkpoint)

    if 'force_field' in hyper_params:
        force_field_parameters = ForceFieldParameters(**hyper_params["force_field"])
        if force_field_parameters.radial_cutoff > 0.0:
            logger.info("Augmenting the AXL_network with an excluding Force Field.")
            axl_network = ForceFieldAugmentedScoreNetwork(axl_network, force_field_parameters)
        else:
            logger.info("Force field parameters are present, but the radial cutoff is zero. "
                        "Using original AXL network")

    logger.info("Instantiate generator...")
    trajectory_initializer = instantiate_trajectory_initializer(
        sampling_parameters=sampling_parameters,
        path_to_constraint_data_pickle=args.path_to_constraint_data_pickle)

    generator = instantiate_generator(
        sampling_parameters=sampling_parameters,
        noise_parameters=noise_parameters,
        axl_network=axl_network,
        trajectory_initializer=trajectory_initializer
    )

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

    sample_energies = None
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

    # If Orion is on, report something to tell Orion the calculation is done.
    if orion.client.cli.IS_ORION_ON:
        if oracle_parameters:
            logger.info("Reporting largest sample energy to ORION.")
            results = dict(name='maximum_oracle_energy', type="objective", value=sample_energies.max().item())
            orion.client.report_results([results])
        else:
            logger.info("Reporting dummy results to ORION.")
            results = dict(name='dummy', type="objective", value=0.0)
            orion.client.report_results([results])

    logger.info("Done!")


if __name__ == "__main__":
    EXP_PATH = Path("/Users/simonblackburn/projects/courtois2024/experiments/lattice_diffusion_experiments/experiments_with_tanh")
    CONFIG_PATH = EXP_PATH / "sampling_config.yaml"
    CKPT_PATH = EXP_PATH / "output/last_model/last_model-epoch=145-step=014600.ckpt"
    OUTPUT_PATH = EXP_PATH / "trajectories"
    args = [
        "--config",
        str(CONFIG_PATH),
        "--checkpoint",
        str(CKPT_PATH),
        "--output",
        str(OUTPUT_PATH),
        "--device",
        "cpu",
    ]
    main(args)

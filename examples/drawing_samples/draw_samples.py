"""Draw Samples.

This script draws samples from a checkpoint.

THIS SCRIPT IS AN EXAMPLE. IT SHOULD BE MODIFIED DEPENDING ON USER PREFERENCES.
"""
import logging
from pathlib import Path

import numpy as np
import torch

from crystal_diffusion.generators.instantiate_generator import \
    instantiate_generator
from crystal_diffusion.generators.predictor_corrector_position_generator import \
    PredictorCorrectorSamplingParameters
from crystal_diffusion.models.position_diffusion_lightning_model import \
    PositionDiffusionLightningModel
from crystal_diffusion.oracle.energies import compute_oracle_energies
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.samples.sampling import create_batch_of_samples
from crystal_diffusion.utils.logging_utils import setup_analysis_logger

logger = logging.getLogger(__name__)
setup_analysis_logger()

checkpoint_path = ("/network/scratch/r/rousseab/experiments/sept21_egnn_2x2x2/run4/"
                   "output/best_model/best_model-epoch=024-step=019550.ckpt")
samples_dir = Path(
    "/network/scratch/r/rousseab/experiments/sept21_egnn_2x2x2/run4_samples/samples"
)
samples_dir.mkdir(exist_ok=True)

device = torch.device("cuda")


spatial_dimension = 3
number_of_atoms = 64
atom_types = np.ones(number_of_atoms, dtype=int)

acell = 10.86
box = np.diag([acell, acell, acell])

number_of_samples = 128
total_time_steps = 1000
number_of_corrector_steps = 1

noise_parameters = NoiseParameters(
    total_time_steps=total_time_steps,
    corrector_step_epsilon=2e-7,
    sigma_min=0.0001,
    sigma_max=0.2,
)

sampling_parameters = PredictorCorrectorSamplingParameters(
    number_of_corrector_steps=number_of_corrector_steps,
    spatial_dimension=spatial_dimension,
    number_of_atoms=number_of_atoms,
    number_of_samples=number_of_samples,
    cell_dimensions=[acell, acell, acell],
    record_samples=True,
)


if __name__ == "__main__":
    logger.info("Loading checkpoint...")
    pl_model = PositionDiffusionLightningModel.load_from_checkpoint(checkpoint_path)
    pl_model.eval()

    sigma_normalized_score_network = pl_model.sigma_normalized_score_network

    logger.info("Instantiate generator...")
    position_generator = instantiate_generator(
        sampling_parameters=sampling_parameters,
        noise_parameters=noise_parameters,
        sigma_normalized_score_network=sigma_normalized_score_network,
    )

    logger.info("Drawing samples...")
    with torch.no_grad():
        samples_batch = create_batch_of_samples(
            generator=position_generator,
            sampling_parameters=sampling_parameters,
            device=device,
        )

    sample_output_path = str(samples_dir / "diffusion_samples.pt")
    position_generator.sample_trajectory_recorder.write_to_pickle(sample_output_path)
    logger.info("Done Generating Samples")

    logger.info("Compute energy from Oracle")
    sample_energies = compute_oracle_energies(samples_batch)

    energy_output_path = str(samples_dir / "diffusion_energies.pt")
    with open(energy_output_path, "wb") as fd:
        torch.save(sample_energies, fd)

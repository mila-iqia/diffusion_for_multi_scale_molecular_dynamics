"""Draw Langevin Samples


This script draws samples from a checkpoint using the Langevin sampler.
"""
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch

from crystal_diffusion import DATA_DIR
from crystal_diffusion.generators.langevin_generator import LangevinGenerator
from crystal_diffusion.generators.predictor_corrector_position_generator import \
    PredictorCorrectorSamplingParameters
from crystal_diffusion.models.position_diffusion_lightning_model import PositionDiffusionLightningModel
from crystal_diffusion.oracle.lammps import get_energy_and_forces_from_lammps
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.logging_utils import setup_analysis_logger

logger = logging.getLogger(__name__)
setup_analysis_logger()

checkpoint_path = '/network/scratch/r/rousseab/checkpoints/EGNN_Sept_10/last_model-epoch=045-step=035972.ckpt'   


samples_dir = Path("/network/scratch/r/rousseab/samples_EGNN_Sept_10_tight_sigmas/SDE/samples_v1")
samples_dir.mkdir(exist_ok=True) 

device = torch.device('cuda')


spatial_dimension = 3
number_of_atoms = 64
atom_types = np.ones(number_of_atoms, dtype=int)

acell = 10.86
box = np.diag([acell, acell, acell])

number_of_samples = 32
total_time_steps = 200
number_of_corrector_steps = 10

noise_parameters = NoiseParameters(total_time_steps=total_time_steps,
                                   corrector_step_epsilon=2e-7,
                                   sigma_min=0.02,
                                   sigma_max=0.2)


pc_sampling_parameters = PredictorCorrectorSamplingParameters(
        number_of_corrector_steps=number_of_corrector_steps, 
        spatial_dimension=spatial_dimension, 
        number_of_atoms=number_of_atoms, 
        number_of_samples=number_of_samples, 
        cell_dimensions=[acell, acell, acell], 
        record_samples=True)


if __name__ == '__main__':

    pl_model = PositionDiffusionLightningModel.load_from_checkpoint(checkpoint_path)
    pl_model.eval()

    sigma_normalized_score_network = pl_model.sigma_normalized_score_network

    position_generator = LangevinGenerator(noise_parameters=noise_parameters, 
                                           sampling_parameters=pc_sampling_parameters, 
                                           sigma_normalized_score_network=sigma_normalized_score_network)


    # Draw some samples, create some plots
    unit_cells = torch.Tensor(box).repeat(number_of_samples, 1, 1).to(device)

    logger.info("Drawing samples")
    with torch.no_grad():
        samples = position_generator.sample(number_of_samples=number_of_samples,
                                            device=device,
                                            unit_cell=unit_cells)


    sample_output_path = str(samples_dir / "diffusion_position_SDE_samples.pt")
    position_generator.sample_trajectory_recorder.write_to_pickle(sample_output_path)
    logger.info("Done Generating Samples")


    batch_relative_positions = samples.cpu().numpy()
    batch_positions = np.dot(batch_relative_positions, box)

    list_energy = []
    logger.info("Compute energy from Oracle")
    with tempfile.TemporaryDirectory() as lammps_work_directory:
        for idx, positions in enumerate(batch_positions):
            energy, forces = get_energy_and_forces_from_lammps(positions,
                                                               box,
                                                               atom_types,
                                                               tmp_work_dir=lammps_work_directory,
                                                               pair_coeff_dir=DATA_DIR)
            list_energy.append(energy)
    energies = torch.tensor(list_energy)

    energy_output_path = str(samples_dir / f"diffusion_energies_Langevin_samples.pt")
    with open(energy_output_path, "wb") as fd:
        torch.save(energies, fd)

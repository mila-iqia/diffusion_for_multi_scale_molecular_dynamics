"""Sampling Si diffusion 1x1x1.

This script loads a pre-trained model, creates the sampler and draws samples.
The energy of the samples are then obtained from the LAMMPS oracle. The dump
files are renamed and kept so that they can be visualized.
"""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from yaml import load

from crystal_diffusion import DATA_DIR, TOP_DIR
from crystal_diffusion.generators.predictor_corrector_position_sampler import \
    AnnealedLangevinDynamicsSampler
from crystal_diffusion.models.model_loader import load_diffusion_model
from crystal_diffusion.oracle.lammps import get_energy_and_forces_from_lammps
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.logging_utils import setup_analysis_logger

logger = logging.getLogger(__name__)

setup_analysis_logger()

dataset_name = 'si_diffusion_2x2x2'
# dataset_name = 'si_diffusion_1x1x1'


# TODO: cell_size should be easily available programmatically.
if dataset_name == 'si_diffusion_1x1x1':
    cell_size = 5.43
    experiment_dir = TOP_DIR.joinpath("experiments/si_diffusion_1x1x1/run1/")
    config_path = str(experiment_dir.joinpath("config_diffusion.yaml"))
    checkpoint_path = str(experiment_dir.joinpath("best_model/best_model-epoch=156-step=061386.ckpt"))
elif dataset_name == 'si_diffusion_2x2x2':
    cell_size = 10.86
    experiment_dir = TOP_DIR.joinpath("experiments/si_diffusion_2x2x2/run1/")
    config_path = str(experiment_dir.joinpath("config_diffusion.yaml"))
    checkpoint_path = str(experiment_dir.joinpath("best_model/best_model-epoch=1972-step=173623.ckpt"))


lammps_work_directory = Path(__file__).parent.joinpath(f"lammps_work_directory/{dataset_name}")
lammps_work_directory.mkdir(exist_ok=True, parents=True)
lammps_work_directory = str(lammps_work_directory)

number_of_corrector_steps = 3
total_time_steps = 100
noise_parameters = NoiseParameters(total_time_steps=total_time_steps)

box = np.diag([cell_size, cell_size, cell_size])

number_of_samples = 4096


if __name__ == '__main__':
    logger.info("Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    with open(config_path, 'r') as stream:
        hyper_params = load(stream, Loader=yaml.FullLoader)
    pl_model = load_diffusion_model(hyper_params)
    pl_model.load_state_dict(state_dict=checkpoint['state_dict'])

    score_network_parameters = pl_model.hyper_params.score_network_parameters
    number_of_atoms = score_network_parameters.number_of_atoms
    atom_types = np.ones(number_of_atoms, dtype=int)

    sigma_normalized_score_network = pl_model.sigma_normalized_score_network

    logger.info("Creating sampler")
    pc_sampler = AnnealedLangevinDynamicsSampler(noise_parameters=noise_parameters,
                                                 number_of_corrector_steps=number_of_corrector_steps,
                                                 number_of_atoms=number_of_atoms,
                                                 spatial_dimension=score_network_parameters.spatial_dimension,
                                                 sigma_normalized_score_network=sigma_normalized_score_network)

    logger.info("Draw samples")
    samples = pc_sampler.sample(number_of_samples)

    batch_relative_positions = samples.cpu().numpy()
    batch_positions = np.dot(batch_relative_positions, box)

    list_energy = []

    logger.info("Compute energy from Oracle")
    for idx, positions in enumerate(batch_positions):
        energy, forces = get_energy_and_forces_from_lammps(positions,
                                                           box,
                                                           atom_types,
                                                           tmp_work_dir=lammps_work_directory,
                                                           pair_coeff_dir=DATA_DIR)
        list_energy.append(energy)
        src = os.path.join(lammps_work_directory, "dump.yaml")
        dst = os.path.join(lammps_work_directory, f"dump_{idx}.yaml")
        os.rename(src, dst)

    df = pd.DataFrame({'energy': list_energy})
    pd.to_pickle(df, f'{dataset_name}_energy_samples.pkl')

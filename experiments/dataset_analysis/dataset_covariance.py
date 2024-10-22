"""Effective Dataset Variance.

The goal of this script is to compute the effective "sigma_d" of the
actual datasets, that is, the standard deviation of the displacement
from equilibrium, in fractional coordinates.
"""
import logging

import einops
import torch
from crystal_diffusion import ANALYSIS_RESULTS_DIR, DATA_DIR
from crystal_diffusion.data.diffusion.data_loader import (
    LammpsForDiffusionDataModule, LammpsLoaderParameters)
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from crystal_diffusion.utils.logging_utils import setup_analysis_logger
from tqdm import tqdm

logger = logging.getLogger(__name__)
dataset_name = 'si_diffusion_2x2x2'
# dataset_name = 'si_diffusion_1x1x1'

output_dir = ANALYSIS_RESULTS_DIR / "covariances"
output_dir.mkdir(exist_ok=True)


if dataset_name == 'si_diffusion_1x1x1':
    max_atom = 8
    translation = torch.tensor([0.125, 0.125, 0.125])
elif dataset_name == 'si_diffusion_2x2x2':
    max_atom = 64
    translation = torch.tensor([0.0625, 0.0625, 0.0625])

lammps_run_dir = DATA_DIR / dataset_name
processed_dataset_dir = lammps_run_dir / "processed"

cache_dir = lammps_run_dir / "cache"

data_params = LammpsLoaderParameters(batch_size=2048, max_atom=max_atom)

if __name__ == '__main__':
    setup_analysis_logger()
    logger.info(f"Computing the covariance matrix for {dataset_name}")

    datamodule = LammpsForDiffusionDataModule(
        lammps_run_dir=lammps_run_dir,
        processed_dataset_dir=processed_dataset_dir,
        hyper_params=data_params,
        working_cache_dir=cache_dir,
    )
    datamodule.setup()

    train_dataset = datamodule.train_dataset

    list_means = []
    for batch in tqdm(datamodule.train_dataloader(), "Mean"):
        x = map_relative_coordinates_to_unit_cell(batch['relative_coordinates'] + translation)
        list_means.append(x.mean(dim=0))

    # Drop the last batch, which might not have dimension batch_size
    x0 = torch.stack(list_means[:-1]).mean(dim=0)

    list_covariances = []
    list_sizes = []
    for batch in tqdm(datamodule.train_dataloader(), "displacements"):
        x = map_relative_coordinates_to_unit_cell(batch['relative_coordinates'] + translation)
        list_sizes.append(x.shape[0])
        displacements = einops.rearrange(x - x0, "batch natoms space -> batch (natoms space)")
        covariance = (displacements[:, None, :] * displacements[:, :, None]).sum(dim=0)
        list_covariances.append(covariance)

    covariance = torch.stack(list_covariances).sum(dim=0) / sum(list_sizes)

    output_file = output_dir / f"covariance_{dataset_name}.pkl"
    logger.info(f"Writing to file {output_file}...")
    with open(output_file, 'wb') as fd:
        torch.save(covariance, fd)

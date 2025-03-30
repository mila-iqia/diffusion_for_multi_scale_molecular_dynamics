"""Effective Dataset Variance.

The goal of this script is to compute the effective "sigma_d" of the
actual datasets, that is, the standard deviation of the displacement
from equilibrium, in fractional coordinates.
"""
import logging

import einops
import torch
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics import DATA_DIR
from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import (
    LammpsDataModuleParameters, LammpsForDiffusionDataModule)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    setup_analysis_logger
from experiments.dataset_analysis import RESULTS_DIR


def get_data_module(dataset_name: str):
    """Convenience method to get the data module."""
    match dataset_name:
        case "si_diffusion_1x1x1":
            max_atom = 8
        case "si_diffusion_2x2x2":
            max_atom = 64
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    lammps_run_dir = DATA_DIR / dataset_name
    processed_dataset_dir = lammps_run_dir / "processed"

    cache_dir = lammps_run_dir / "cache"

    data_params = LammpsDataModuleParameters(batch_size=2048,
                                             max_atom=max_atom,
                                             noise_parameters=NoiseParameters(total_time_steps=1),
                                             use_optimal_transport=False,
                                             use_fixed_lattice_parameters=True,
                                             elements=['Si'])
    datamodule = LammpsForDiffusionDataModule(
        lammps_run_dir=lammps_run_dir,
        processed_dataset_dir=processed_dataset_dir,
        hyper_params=data_params,
        working_cache_dir=cache_dir,
    )
    datamodule.setup()

    return datamodule


logger = logging.getLogger(__name__)
dataset_name = "si_diffusion_2x2x2"
# dataset_name = 'si_diffusion_1x1x1'

output_dir = RESULTS_DIR / "covariances"
output_dir.mkdir(exist_ok=True)

if __name__ == "__main__":
    setup_analysis_logger()
    logger.info(f"Computing the covariance matrix for {dataset_name}")
    datamodule = get_data_module(dataset_name)
    match dataset_name:
        case "si_diffusion_1x1x1":
            translation = torch.tensor([0.125, 0.125, 0.125])
        case "si_diffusion_2x2x2":
            translation = torch.tensor([0.0625, 0.0625, 0.0625])
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    train_dataset = datamodule.train_dataset

    list_means = []
    for batch in tqdm(datamodule.train_dataloader(), "Mean"):
        x = map_relative_coordinates_to_unit_cell(
            batch["relative_coordinates"] + translation
        )
        list_means.append(x.mean(dim=0))

    # Drop the last batch, which might not have dimension batch_size
    x0 = torch.stack(list_means[:-1]).mean(dim=0)

    list_covariances = []
    list_sizes = []
    for batch in tqdm(datamodule.train_dataloader(), "displacements"):
        x = map_relative_coordinates_to_unit_cell(
            batch["relative_coordinates"] + translation
        )
        list_sizes.append(x.shape[0])
        displacements = einops.rearrange(
            x - x0, "batch natoms space -> batch (natoms space)"
        )
        covariance = (displacements[:, None, :] * displacements[:, :, None]).sum(dim=0)
        list_covariances.append(covariance)

    covariance = torch.stack(list_covariances).sum(dim=0) / sum(list_sizes)

    output_file = output_dir / f"covariance_{dataset_name}.pkl"
    logger.info(f"Writing to file {output_file}...")
    with open(output_file, "wb") as fd:
        torch.save(covariance, fd)

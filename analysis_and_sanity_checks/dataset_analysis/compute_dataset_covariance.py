"""Effective Dataset Variance.

The goal of this script is to compute the effective "sigma_d" of the
actual datasets, that is, the standard deviation of the displacement
from equilibrium, in fractional coordinates.
"""
import logging

import einops
import torch
from tqdm import tqdm
from utilities import RESULTS_DIR
from utilities.data import get_data_module

from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell

logger = logging.getLogger(__name__)

list_dataset_names = ["Si_diffusion_1x1x1", "Si_diffusion_2x2x2", "Si_diffusion_3x3x3"]

# A translation will be applied to avoid edge effects.
list_translations = [torch.tensor([0.125, 0.125, 0.125]),
                     torch.tensor([0.0625, 0.0625, 0.0625]),
                     torch.tensor([0.041667, 0.041667, 0.041667])]

if __name__ == "__main__":

    for dataset_name, translation in zip(list_dataset_names, list_translations):
        logger.info(f"Computing the covariance matrix for {dataset_name}")
        datamodule = get_data_module(dataset_name)

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

        output_file = RESULTS_DIR / f"covariance_{dataset_name}.pkl"
        logger.info(f"Writing to file {output_file}...")
        with open(output_file, "wb") as fd:
            torch.save(covariance, fd)

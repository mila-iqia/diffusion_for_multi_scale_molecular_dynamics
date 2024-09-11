"""Convert Si data to flowMM format.

This script creates csv files in the format expected by flowMM so we can test it
on our own dataset.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from crystal_diffusion import DATA_DIR
from crystal_diffusion.data.diffusion.data_loader import (
    LammpsForDiffusionDataModule, LammpsLoaderParameters)
from crystal_diffusion.utils.structure_utils import create_structure

flowmm_data_path = Path("/Users/bruno/PycharmProjects/flowmm/data/carbon_24/")

top_dir = Path(
    "/Users/bruno/PycharmProjects/diffusion_for_multi_scale_molecular_dynamics/sanity_checks/data_for_flowmm/"
)

kind = "1x1x1"

if kind == "2x2x2":
    dataset_name = "si_diffusion_2x2x2"
    here = top_dir / "si_2x2x2_bruno"
    data_params = LammpsLoaderParameters(batch_size=64, max_atom=64)
    train_size = 16384
    val_size = 1024
    test_size = 1024
elif kind == "1x1x1":
    dataset_name = "si_diffusion_1x1x1"
    here = top_dir / "si_1x1x1_bruno"
    data_params = LammpsLoaderParameters(batch_size=64, max_atom=8)
    train_size = 1024
    val_size = 128
    test_size = 128

here.mkdir(exist_ok=True)

lammps_run_dir = str(DATA_DIR / dataset_name)
processed_dataset_dir = str(DATA_DIR / dataset_name / "processed")

cache_dir = str(DATA_DIR / dataset_name / "cache")

if __name__ == "__main__":
    flowmm_df = pd.read_csv(flowmm_data_path / "train.csv")

    row = flowmm_df.iloc[0]

    datamodule = LammpsForDiffusionDataModule(
        lammps_run_dir=lammps_run_dir,
        processed_dataset_dir=processed_dataset_dir,
        hyper_params=data_params,
        working_cache_dir=cache_dir,
    )
    datamodule.setup()

    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.valid_dataset

    list_datasets = [train_dataset, val_dataset, val_dataset]
    list_names = ["train", "val", "test"]
    list_sizes = [train_size, val_size, test_size]
    list_offsets = [0, 0, val_size]

    for name, dataset, size, offset in zip(
        list_names, list_datasets, list_sizes, list_offsets
    ):
        list_rows = []
        for i in tqdm(range(size), name):
            row = dict()
            data = dataset[offset + i]
            natom = int(data["natom"])

            row["energy_per_atom"] = data["potential_energy"].numpy() / natom
            row["material_id"] = f"{name}_id_{i}"

            basis_vectors = np.diag(data["box"].numpy())
            relative_coordinates = data["relative_coordinates"].numpy()
            species = natom * ["Si"]

            structure = create_structure(basis_vectors, relative_coordinates, species)
            row["cif"] = structure.to(fmt="cif")
            list_rows.append(row)

        df = pd.DataFrame(list_rows)
        df.to_csv(here / f"{name}.csv")

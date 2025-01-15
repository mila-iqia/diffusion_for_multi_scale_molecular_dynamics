import logging
from dataclasses import dataclass
from typing import List, Optional

import einops
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.data_module_parameters import \
    DataModuleParameters
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, CARTESIAN_FORCES, RELATIVE_COORDINATES)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class GaussianDataModuleParameters(DataModuleParameters):
    """Hyper-parameters for a Gaussian, in memory data module."""
    data_source = "gaussian"

    random_seed: int
    # the number of atoms in a configuration.
    number_of_atoms: int

    # Standard deviation of the Gaussian distribution.
    # The covariance matrix is assumed to be proportional to the identity matrix.
    sigma_d: float = 0.01

    # mean of the Gaussian Distribution
    equilibrium_relative_coordinates: List[List[float]]

    train_dataset_size: int = 8_192
    valid_dataset_size: int = 1_024

    def __post_init__(self):
        """Post init."""
        assert self.sigma_d > 0.0, "the sigma_d parameter should be positive."

        assert len(self.equilibrium_relative_coordinates) == self.number_of_atoms, \
            "There should be exactly one list of equilibrium coordinates per atom."

        for x in self.equilibrium_relative_coordinates:
            assert len(x) == self.spatial_dimension, \
                "The equilibrium coordinates should be consistent with the spatial dimension."

        assert len(self.elements) == 1, "There can only be one element type for the gaussian data module."


class GaussianDataModule(pl.LightningDataModule):
    """Gaussian Data Module.

    Data module class that creates an in-memory dataset of relative coordinates that follow a Gaussian distribution.
    """

    def __init__(
        self,
        hyper_params: GaussianDataModuleParameters,
    ):
        """Init method."""
        super().__init__()

        self.random_seed = hyper_params.random_seed
        self.number_of_atoms = hyper_params.number_of_atoms
        self.spatial_dimension = hyper_params.spatial_dimension
        self.sigma_d = hyper_params.sigma_d
        self.equilibrium_coordinates = torch.tensor(hyper_params.equilibrium_relative_coordinates,
                                                    dtype=torch.float)

        self.train_dataset_size = hyper_params.train_dataset_size
        self.valid_dataset_size = hyper_params.valid_dataset_size

        assert hyper_params.batch_size, "batch_size must be specified"

        self.batch_size = hyper_params.batch_size
        self.train_size = hyper_params.train_batch_size
        self.valid_size = hyper_params.valid_batch_size

        self.num_workers = hyper_params.num_workers

        self.element_types = ElementTypes(hyper_params.elements)

    def setup(self, stage: Optional[str] = None):
        """Setup method."""
        self.train_dataset = []
        self.valid_dataset = []

        rng = torch.Generator()
        rng.manual_seed(self.random_seed)

        box = torch.ones(self.spatial_dimension, dtype=torch.float)

        atom_types = torch.zeros(self.number_of_atoms, dtype=torch.long)

        for dataset, batch_size in zip([self.train_dataset, self.valid_dataset],
                                       [self.train_dataset_size, self.valid_dataset_size]):

            mean = einops.repeat(self.equilibrium_coordinates,
                                 "natoms space -> batch natoms space", batch=batch_size)
            std = self.sigma_d * torch.ones_like(mean)
            relative_coordinates = map_relative_coordinates_to_unit_cell(
                torch.normal(mean=mean, std=std, generator=rng).to(torch.float))

            for x in relative_coordinates:
                row = {
                    "natom": self.number_of_atoms,
                    "box": box,
                    RELATIVE_COORDINATES: x,
                    ATOM_TYPES: atom_types,
                    CARTESIAN_FORCES: torch.zeros_like(x),
                    "potential_energy": torch.tensor([0.0], dtype=torch.float),
                }
                dataset.append(row)

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader using the training data parser."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Create the validation dataloader using the validation data parser."""
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Creates the testing dataloader using the testing data parser."""
        raise NotImplementedError("Test set is not defined at the moment.")

    def clean_up(self):
        """Nothing to clean."""
        pass

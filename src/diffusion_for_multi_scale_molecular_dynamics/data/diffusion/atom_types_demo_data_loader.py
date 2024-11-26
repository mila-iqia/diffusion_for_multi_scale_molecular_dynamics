"""DataLoader from LAMMPS outputs for a diffusion model."""

import logging
import typing
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional

import datasets
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.data_preprocess import \
    LammpsProcessorForDiffusion
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, CARTESIAN_FORCES, CARTESIAN_POSITIONS, RELATIVE_COORDINATES)

logger = logging.getLogger(__name__)


class AtomTypesDemoDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        num_atom_types: int,
        num_atoms: int,
        spatial_dimension: int = 3,
        use_physical_positions: bool = False
    ):
        self.num_samples = num_samples
        self.num_atom_types = num_atom_types
        self.num_atoms = num_atoms
        self.spatial_dimension = spatial_dimension
        self.use_physical_positions = use_physical_positions

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = {}
        for var in [CARTESIAN_POSITIONS, RELATIVE_COORDINATES, CARTESIAN_FORCES]:
            x[var] = torch.zeros(self.num_atoms, self.spatial_dimension)
        if self.use_physical_positions:
            x[RELATIVE_COORDINATES][:, 0] = torch.linspace(0, 1, self.num_atoms + 1)[:-1].view(1, -1)
            x[CARTESIAN_POSITIONS] = x[RELATIVE_COORDINATES] * 10
        atom_types = torch.arange(idx, idx + self.num_atoms) % self.num_atom_types
        x[ATOM_TYPES] = atom_types
        x["box"] = torch.ones(self.spatial_dimension) * 10.0
        x["potential_energy"] = torch.zeros(1)
        return x


@dataclass(kw_only=True)
class AtomTypesDemoLoaderParameters:
    """Hyper parameters for AtomTypesDemo dataset.

    This dataset creates a chain of N atoms with sequential atom types.
    0 - 1 - 2 - 3 - 0 - 1 - 2 - 3 ...
    The reduced coordinates of all atoms are set to 0 because they are irrelevant. The first atom can be any type,
    chosen randomly. It fixes the rest of the chain. This is meant to be used with an MLP that is not permutation
    equivariant.
    """

    # Either batch_size XOR train_batch_size and valid_batch_size should be specified.
    batch_size: Optional[int] = None
    train_batch_size: Optional[int] = None
    valid_batch_size: Optional[int] = None
    num_workers: int = 0
    max_atom: int = 64  # also acts as the chain length
    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live. Irrelevant.
    use_physical_positions: bool = False
    num_atom_types: int = 4  # max atom types.
    train_num_samples: int = 1024
    valid_num_samples: int = 512


class AtomTypesDataModule(pl.LightningDataModule):
    """Data module class that prepares dataset parsers and instantiates data loaders."""

    def __init__(
        self,
        hyper_params: AtomTypesDemoLoaderParameters,
    ):
        """Initialize a dataset of LAMMPS structures for training a diffusion model.

        Args:
            hyper_params: hyperparameters
        """
        super().__init__()
        self.num_workers = hyper_params.num_workers
        self.max_atom = hyper_params.max_atom  # number of atoms to pad tensors
        self.spatial_dim = hyper_params.spatial_dimension
        self.use_physical_positions = hyper_params.use_physical_positions
        self.num_atom_types = hyper_params.num_atom_types
        self.train_num_samples = hyper_params.train_num_samples
        self.valid_num_samples = hyper_params.valid_num_samples

        if hyper_params.batch_size is None:
            assert (
                hyper_params.valid_batch_size is not None
            ), "If batch_size is None, valid_batch_size must be specified."
            assert (
                hyper_params.train_batch_size is not None
            ), "If batch_size is None, train_batch_size must be specified."

            self.train_batch_size = hyper_params.train_batch_size
            self.valid_batch_size = hyper_params.valid_batch_size

        else:
            assert (
                hyper_params.valid_batch_size is None
            ), "If batch_size is specified, valid_batch_size must be None."
            assert (
                hyper_params.train_batch_size is None
            ), "If batch_size is specified, train_batch_size must be None."
            self.train_batch_size = hyper_params.batch_size
            self.valid_batch_size = hyper_params.batch_size

    def setup(self, stage: Optional[str] = None):
        """Parse and split all samples across the train/valid/test parsers."""
        # here, we will actually assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = AtomTypesDemoDataset(
                num_samples=self.train_num_samples,
                num_atom_types=self.num_atom_types,
                num_atoms=self.max_atom,
                spatial_dimension=self.spatial_dim,
                use_physical_positions=self.use_physical_positions
            )

            self.valid_dataset = AtomTypesDemoDataset(
                num_samples=self.valid_num_samples,
                num_atom_types=self.num_atom_types,
                num_atoms=self.max_atom,
                spatial_dimension=self.spatial_dim,
                use_physical_positions=self.use_physical_positions
            )
        else:
            raise NotImplementedError("Test mode needs to be implemented.")

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader using the training data parser."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Create the validation dataloader using the validation data parser."""
        return DataLoader(
            self.valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Creates the testing dataloader using the testing data parser."""
        raise NotImplementedError("Test set is not defined at the moment.")

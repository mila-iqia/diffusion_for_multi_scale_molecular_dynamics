"""DataLoader from LAMMPS outputs for a diffusion model."""

import logging
from dataclasses import dataclass
from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, CARTESIAN_FORCES, CARTESIAN_POSITIONS, RELATIVE_COORDINATES, LATTICE_PARAMETERS)


logger = logging.getLogger(__name__)


class LatticeDemoDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        lattice_averages: List[float],
        lattice_stddev: List[float],
        num_atom_types: int = 1,
        num_atoms: int = 1,
        spatial_dimension: int = 3,
        use_physical_positions: bool = False,
        lattice_vectors_minimum: float = 0.1
    ):
        self.num_samples = num_samples
        self.num_atom_types = num_atom_types
        self.num_atoms = num_atoms
        self.spatial_dimension = spatial_dimension
        self.use_physical_positions = use_physical_positions
        self.lattice_average = torch.tensor(lattice_averages)
        self.lattice_stddev = torch.tensor(lattice_stddev)
        self.lattice_minimum = lattice_vectors_minimum
        self.lattice_scales = self.get_parameters_scale().clip(min=1e-4)

    def get_parameters_scale(self):
        num_params = int(self.spatial_dimension * (self.spatial_dimension + 1) / 2)
        max_params = -torch.ones(num_params) * torch.inf
        for idx in range(self.num_samples):
            lattice_params = self.generate_lattice_parameters(idx)
            max_params = torch.maximum(max_params, lattice_params)
        return max_params

    def generate_lattice_parameters(self, seed: int):
        torch.manual_seed(seed)
        z = torch.randn(self.spatial_dimension)
        lattice_parameters = self.lattice_average + z * self.lattice_stddev
        lattice_parameters = lattice_parameters.clip(min=self.lattice_minimum)
        num_angle_params = int(self.spatial_dimension * (self.spatial_dimension - 1) / 2)
        lattice_parameters = torch.cat([lattice_parameters, torch.zeros(num_angle_params)])
        return lattice_parameters

    def scale_lattice_parameters(self, lattice_parameters: torch.Tensor):
        return lattice_parameters / self.lattice_scales

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
        lattice_parameters = self.generate_lattice_parameters(idx)
        lattice_parameters = self.scale_lattice_parameters(lattice_parameters)
        x[LATTICE_PARAMETERS] = lattice_parameters
        x["potential_energy"] = torch.zeros(1)
        return x


@dataclass(kw_only=True)
class LatticeDemoParameters:
    """Hyper parameters for LatticeDemo dataset.

    This dataset generates samples with orthogonal boxes with the three sizes generated randomly from a gaussian
    distribution.
    """

    # Either batch_size XOR train_batch_size and valid_batch_size should be specified.
    batch_size: Optional[int] = None
    train_batch_size: Optional[int] = None
    valid_batch_size: Optional[int] = None
    num_workers: int = 0
    max_atom: int = 1  # also acts as the chain length
    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live. Irrelevant.
    use_physical_positions: bool = False
    num_atom_types: int = 1  # max atom types.
    lattice_averages: List[float]
    lattice_stddev: List[float]
    train_num_samples: int = 1024
    valid_num_samples: int = 512


class LatticeDemoDataModule(pl.LightningDataModule):
    """Data module class that prepares dataset parsers and instantiates data loaders."""

    def __init__(
        self,
        hyper_params: LatticeDemoParameters,
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
        self.lattice_averages = hyper_params.lattice_averages
        self.lattice_stddev = hyper_params.lattice_stddev
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
            self.train_dataset = LatticeDemoDataset(
                num_samples=self.train_num_samples,
                num_atom_types=self.num_atom_types,
                num_atoms=self.max_atom,
                spatial_dimension=self.spatial_dim,
                use_physical_positions=self.use_physical_positions,
                lattice_averages=self.lattice_averages,
                lattice_stddev=self.lattice_stddev
            )

            self.valid_dataset = LatticeDemoDataset(
                num_samples=self.valid_num_samples,
                num_atom_types=self.num_atom_types,
                num_atoms=self.max_atom,
                spatial_dimension=self.spatial_dim,
                use_physical_positions=self.use_physical_positions,
                lattice_averages=self.lattice_averages,
                lattice_stddev=self.lattice_stddev
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

    def clean_up(self):
        pass

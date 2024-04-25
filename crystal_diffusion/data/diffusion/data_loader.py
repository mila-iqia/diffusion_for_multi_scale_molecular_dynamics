"""DataLoader from LAMMPS outputs for a diffusion model."""
import logging
import typing
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional

import datasets
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crystal_diffusion.data.diffusion.data_preprocess import \
    LammpsProcessorForDiffusion

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class LammpsLoaderParameters:
    """Base Hyper-parameters for score networks."""

    batch_size: int = 64
    num_workers: int = 0
    max_atom: int = 64
    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live.


class LammpsForDiffusionDataModule(pl.LightningDataModule):
    """Data module class that prepares dataset parsers and instantiates data loaders."""

    def __init__(
            self,
            lammps_run_dir: str,
            processed_dataset_dir: str,
            hyper_params: LammpsLoaderParameters,
            working_cache_dir: Optional[str] = None
    ):
        """Initialize a dataset of LAMMPS structures for training a diffusion model.

        Args:
            lammps_run_dir: folder with results from LAMMPS runs,
            processed_dataset_dir: folder where to store processed data,
            hyper_params: hyperparameters
            working_cache_dir (optional): temporary working directory for the Datasets library. If None, the library
                uses the default folder in ~/.cache/. The default path is not cleaned up after a run and can occupy a
                lot of disk space. Defaults to None.
        """
        super().__init__()
        # check_and_log_hp(["batch_size", "num_workers"], hyper_params)  # validate the hyperparameters
        # TODO add the padding parameters for number of atoms
        self.lammps_run_dir = lammps_run_dir
        self.processed_dataset_dir = processed_dataset_dir
        self.working_cache_dir = working_cache_dir
        self.batch_size = hyper_params.batch_size
        self.num_workers = hyper_params.num_workers
        self.max_atom = hyper_params.max_atom  # number of atoms to pad tensors
        self.spatial_dim = hyper_params.spatial_dimension

    @staticmethod
    def dataset_transform(x: Dict[typing.AnyStr, typing.Any], spatial_dim: int = 3) -> Dict[str, torch.Tensor]:
        """Format the tensors for the Datasets library.

        This function is applied right after returning the objects in __getitem__ in a torch DataLoader. Everything is
        already batched. Lightning handles the devices (gpu, cpu, mps, etc.)

        Args:
            x: raw columns from the processed data files. Should contain natom, box, type, position and
                relative_positions.
            spatial_dim (optional): number of spatial dimensions. Defaults to 3.

        Returns:
            transformed_x: formatted values as tensors
        """
        transformed_x = {}
        transformed_x['natom'] = torch.as_tensor(x['natom']).long()  # resulting tensor size: (batchsize, )
        bsize = transformed_x['natom'].size(0)
        transformed_x['box'] = torch.as_tensor(x['box'])  # size: (batchsize, spatial dimension)
        for pos in ['position', 'relative_positions']:
            transformed_x[pos] = torch.as_tensor(x[pos]).view(bsize, -1, spatial_dim)
        transformed_x['type'] = torch.as_tensor(x['type']).long()  # size: (batchsize, max atom)
        transformed_x['potential_energy'] = torch.as_tensor(x['potential_energy'])  # size: (batchsize, )

        return transformed_x

    @staticmethod
    def pad_samples(x: Dict[typing.AnyStr, typing.Any], max_atom: int, spatial_dim: int = 3) -> Dict[str, torch.Tensor]:
        """Pad a sample for batching.

        Args:
            x: initial sample from the dataset. Should contain natom, position, relative_positions and type.
            max_atom: maximum number of atoms to pad to
            spatial_dim (optional): number of spatial dimensions. Defaults to 3.

        Returns:
            x: sample with padded type and position
        """
        natom = x['natom']
        if natom > max_atom:
            raise ValueError(f"Hyper-parameter max_atom is smaller than an example in the dataset with {natom} atoms.")
        x['type'] = F.pad(torch.as_tensor(x['type']).long(), (0, max_atom - natom), 'constant', -1)
        for pos in ['position', 'relative_positions']:
            x[pos] = F.pad(torch.as_tensor(x[pos]).float(), (0, spatial_dim * (max_atom - natom)), 'constant',
                           torch.nan)
        return x

    def setup(self, stage: Optional[str] = None):
        """Parse and split all samples across the train/valid/test parsers."""
        # here, we will actually assign train/val datasets for use in dataloaders
        processed_data = LammpsProcessorForDiffusion(self.lammps_run_dir, self.processed_dataset_dir)

        if stage == "fit" or stage is None:
            self.train_dataset = datasets.Dataset.from_parquet(processed_data.train_files,
                                                               cache_dir=self.working_cache_dir)
            self.valid_dataset = datasets.Dataset.from_parquet(processed_data.valid_files,
                                                               cache_dir=self.working_cache_dir)
            # TODO QoL valid dataset is labelled as train split by Datasets. Find a way to rename.
        else:
            raise NotImplementedError("Test mode needs to be implemented.")
        # TODO test dataset when stage == 'test'
        # we can filter out samples at this stage using the .filter(lambda x: f(x)) with f(x) a boolean function
        # or a .select(list[int]) with a list of indices to keep a subset. This is much faster than .filter
        # padding needs to be done here OR in the preprocessor
        # check if the max number of atoms matches at least the max in the training set
        if max(self.train_dataset['natom']) > self.max_atom:
            raise ValueError(f"Hyper-paramater max_atom {self.max_atom} is smaller than the largest structure in the"
                             + f"dataset which has {max(self.train_dataset['natom'])} atoms.")
        # map() are applied once, not in-place.
        # The keyword argument "batched" can accelerate by working with batches, not useful for padding
        self.train_dataset = self.train_dataset.map(partial(self.pad_samples, max_atom=self.max_atom,
                                                            spatial_dim=self.spatial_dim), batched=False)
        self.valid_dataset = self.valid_dataset.map(partial(self.pad_samples, max_atom=self.max_atom,
                                                            spatial_dim=self.spatial_dim), batched=False)
        # set_transform is applied on-the-fly and is less costly upfront. Works with batches, so we can't use it for
        # padding
        self.train_dataset.set_transform(partial(self.dataset_transform, spatial_dim=self.spatial_dim))
        self.valid_dataset.set_transform(partial(self.dataset_transform, spatial_dim=self.spatial_dim))

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
        """Delete the Datasets working cache."""
        self.train_dataset.cleanup_cache_files()
        self.valid_dataset.cleanup_cache_files()

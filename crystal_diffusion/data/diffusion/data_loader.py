"""DataLoader from LAMMPS outputs for a diffusion model."""
import logging
import typing
from typing import Dict, Optional

import datasets
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from crystal_diffusion.data.diffusion.data_preprocess import \
    LammpsProcessorForDiffusion
from crystal_diffusion.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class LammpsForDiffusionDataModule(pl.LightningDataModule):  # pragma: no cover
    """Data module class that prepares dataset parsers and instantiates data loaders."""

    def __init__(
            self,
            lammps_run_dir: str,
            processed_dataset_dir: str,
            hyper_params: Dict[typing.AnyStr, typing.Any],
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
        check_and_log_hp(["batch_size", "num_workers"], hyper_params)  # validate the hyperparameters
        # TODO add the padding parameters for number of atoms
        self.lammps_run_dir = lammps_run_dir
        self.processed_dataset_dir = processed_dataset_dir
        self.working_cache_dir = working_cache_dir
        self.batch_size = hyper_params["batch_size"]
        self.num_workers = hyper_params["num_workers"]

    @staticmethod
    def dataset_transform(x: Dict[typing.AnyStr, typing.Any]) -> Dict[str, torch.Tensor]:
        """Format the tensors for the Datasets library.

        This function is applied right after returning the objects in __getitem__ in a torch DataLoader. Everything is
        already batched. Lightning handles the devices (gpu, cpu, mps, etc.)

        Args:
            x: raw columns from the processed data files. Should contain natom, box, type and position.

        Returns:
            transformed_x: formatted values as tensors
        """
        transformed_x = {}
        transformed_x['natom'] = torch.as_tensor(x['natom']).long()  # resulting tensor size: (batchsize, )
        bsize = transformed_x['natom'].size(0)
        transformed_x['positions'] = torch.as_tensor(x['position']).view(bsize, -1, 3)  # hard-coding 3D system
        transformed_x['box'] = torch.as_tensor(x['box'])  # size: (batchsize, 3)
        transformed_x['type'] = torch.as_tensor(x['type']).long()  # size: (batchsize, natom after padding)

        return transformed_x

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
        # example for adding arguments to dataset_transform
        # self.train_dataset.set_transform(partial(self.dataset_transform, args=args))
        self.train_dataset.set_transform(self.dataset_transform)
        self.valid_dataset.set_transform(self.dataset_transform)

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
        return DataLoader(
            self.valid_dataset,  # TODO replace with test dataset
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

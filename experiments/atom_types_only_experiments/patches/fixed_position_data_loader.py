import logging
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from equilibrium_structure import create_equilibrium_sige_structure
from torch_geometric.data import DataLoader

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import \
    LammpsLoaderParameters
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, CARTESIAN_FORCES, RELATIVE_COORDINATES)

logger = logging.getLogger(__name__)


class FixedPositionDataModule(pl.LightningDataModule):
    """Data module class that is meant to imitate LammpsForDiffusionDataModule."""

    def __init__(
        self,
        lammps_run_dir: str,  # dummy
        processed_dataset_dir: str,
        hyper_params: LammpsLoaderParameters,
        working_cache_dir: Optional[str] = None,  # dummy
    ):
        """Init method."""
        logger.debug("FixedPositionDataModule!")
        super().__init__()

        assert hyper_params.batch_size, "batch_size must be specified"
        assert hyper_params.train_batch_size, "train_batch_size must be specified"
        assert hyper_params.valid_batch_size, "valid_batch_size must be specified"

        self.batch_size = hyper_params.batch_size
        self.train_size = hyper_params.train_batch_size
        self.valid_size = hyper_params.valid_batch_size

        self.num_workers = hyper_params.num_workers
        self.max_atom = hyper_params.max_atom  # number of atoms to pad tensors

        self.element_types = ElementTypes(hyper_params.elements)

    def setup(self, stage: Optional[str] = None):
        """Setup method."""
        structure = create_equilibrium_sige_structure()

        relative_coordinates = torch.from_numpy(structure.frac_coords).to(torch.float)

        atom_types = torch.tensor(
            [self.element_types.get_element_id(a.name) for a in structure.species]
        )
        box = torch.tensor(structure.lattice.abc)

        row = {
            "natom": len(atom_types),
            "box": box,
            RELATIVE_COORDINATES: relative_coordinates,
            ATOM_TYPES: atom_types,
            CARTESIAN_FORCES: torch.zeros_like(relative_coordinates),
            "potential_energy": 0.0,
        }

        self.train_dataset = [row for _ in range(self.train_size)]
        self.valid_dataset = [row for _ in range(self.valid_size)]

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


if __name__ == "__main__":

    elements = ["Si", "Ge"]
    processed_dataset_dir = Path("/experiments/atom_types_only_experiments")

    hyper_params = LammpsLoaderParameters(
        batch_size=64,
        train_batch_size=1024,
        valid_batch_size=1024,
        num_workers=8,
        max_atom=8,
        elements=elements,
    )

    data_module = FixedPositionDataModule(
        lammps_run_dir="dummy",
        processed_dataset_dir=processed_dataset_dir,
        hyper_params=hyper_params,
    )

    data_module.setup()

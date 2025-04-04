import argparse
import logging
from dataclasses import dataclass, field
from typing import Any, AnyStr, Dict, Optional

import datasets
import pytorch_lightning as pl
import torch
from fixed_position_noising_transform import FixedPositionNoisingTransform

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.data_module_parameters import \
    DataModuleParameters
from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import (
    LammpsDataModuleParameters, LammpsForDiffusionDataModule)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, CARTESIAN_FORCES, CARTESIAN_POSITIONS, LATTICE_PARAMETERS,
    NOISY_ATOM_TYPES, NOISY_LATTICE_PARAMETERS, NOISY_RELATIVE_COORDINATES,
    RELATIVE_COORDINATES)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_unit_cell_to_lattice_parameters
from diffusion_for_multi_scale_molecular_dynamics.utils.reference_configurations import \
    create_equilibrium_sige_structure

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class FixedPositionDataModuleParameters(DataModuleParameters):
    """Hyper-Parameters for a Fixed Position Data module."""

    data_source: str = "FixedPosition"
    noise_parameters: NoiseParameters

    train_dataset_size: int = 1_024
    valid_dataset_size: int = 256

    num_workers: int = 0
    max_atom: int = 8
    spatial_dimension: int = 3
    use_fixed_lattice_parameters: bool = True
    elements: list[str] = field(default_factory=lambda: ["Si", "Ge"])

    def __post_init__(self):
        """Post init."""
        global_error_message = (
            "This is a highly contrived Data Module to demonstrate atom type diffusion. "
            "It is assumed that the system is SiGe with 8 atoms. "
            "Do not change the default input values."
        )
        assert self.num_workers == 0, "num_workers must be 0." + global_error_message
        assert self.max_atom == 8, "max_atoms must be 8." + global_error_message
        assert self.spatial_dimension == 3, (
            "spatial_dimension must be 3." + global_error_message
        )
        assert self.use_fixed_lattice_parameters, (
            "use_fixed_lattice_parameters must be True." + global_error_message
        )
        assert self.elements == ["Si", "Ge"], (
            "elements must be Si and Ge." + global_error_message
        )


class FixedPositionDataModule(LammpsForDiffusionDataModule):
    """Data module class that is meant to imitate LammpsForDiffusionDataModule, but with fixed positions."""

    def __init__(self, hyper_params: FixedPositionDataModuleParameters):
        """Init method."""
        base_class_hyperparams = LammpsDataModuleParameters(
            noise_parameters=hyper_params.noise_parameters,
            use_optimal_transport=False,
            batch_size=hyper_params.batch_size,
            num_workers=hyper_params.num_workers,
            max_atom=hyper_params.max_atom,
            spatial_dimension=hyper_params.spatial_dimension,
            use_fixed_lattice_parameters=True,
            elements=hyper_params.elements,
        )

        self.train_dataset_size = hyper_params.train_dataset_size
        self.valid_dataset_size = hyper_params.valid_dataset_size

        logger.debug("FixedPositionDataModule!")
        super().__init__(
            lammps_run_dir="/dummy/",
            processed_dataset_dir="/dummy/",
            hyper_params=base_class_hyperparams,
            working_cache_dir=None,
        )

        num_atom_types = len(hyper_params.elements)

        # Overload the noising tranform with something that only affects atom types.
        self.noising_transform = FixedPositionNoisingTransform(
            noise_parameters=hyper_params.noise_parameters,
            num_atom_types=num_atom_types,
            spatial_dimension=self.spatial_dim,
        )

    def setup(self, stage: Optional[str] = None):
        """Setup method."""
        structure = create_equilibrium_sige_structure()

        relative_coordinates = torch.from_numpy(structure.frac_coords).to(torch.float)
        cartesian_positions = torch.from_numpy(structure.cart_coords).to(torch.float)

        elements = [a.name for a in structure.species]
        box = torch.tensor(structure.lattice.abc)

        basis_vectors = torch.from_numpy(1.0 * structure.lattice.matrix).to(torch.float)

        lattice_parameters = map_unit_cell_to_lattice_parameters(basis_vectors)

        row = {
            "natom": len(elements),
            "box": box,
            "element": elements,
            CARTESIAN_POSITIONS: cartesian_positions,
            RELATIVE_COORDINATES: relative_coordinates,
            CARTESIAN_FORCES: torch.zeros_like(relative_coordinates),
            LATTICE_PARAMETERS: lattice_parameters,
            "potential_energy": 0.0,
        }

        self.train_dataset = datasets.Dataset.from_list(self.train_dataset_size * [row])
        self.valid_dataset = datasets.Dataset.from_list(self.valid_dataset_size * [row])

        # set_transform is applied on-the-fly and is less costly upfront.
        # Works with batches.
        transform = self.create_composed_transform()
        self.train_dataset.set_transform(transform)
        self.valid_dataset.set_transform(transform)

    def clean_up(self):
        """Nothing to clean."""
        pass


def load_fixed_position_data_module(
    hyper_params: Dict[AnyStr, Any], args: argparse.Namespace
) -> pl.LightningDataModule:
    """Load fixed position data module.

    This method creates the fixed position data module based on configuration and input arguments.

    Args:
        hyper_params: configuration parameters.
        args: parsed command line arguments.

    Returns:
        data_module:  the data module corresponding to the configuration and input arguments.
    """
    assert (
        "data" in hyper_params
    ), "The configuration should contain a 'data' block describing the data source."

    data_config = hyper_params["data"]
    noise = data_config.pop("noise")
    noise_parameters = NoiseParameters(**noise)
    data_params = FixedPositionDataModuleParameters(
        **data_config, noise_parameters=noise_parameters
    )

    data_module = FixedPositionDataModule(hyper_params=data_params)
    return data_module


if __name__ == "__main__":
    noise_parameters = NoiseParameters(total_time_steps=100)

    hyper_params = FixedPositionDataModuleParameters(
        batch_size=8,
        noise_parameters=noise_parameters,
        train_dataset_size=32,
        valid_dataset_size=16,
    )

    data_module = FixedPositionDataModule(hyper_params=hyper_params)
    data_module.setup()

    # Sanity Check: test that only the atom types are noised.
    for dataloader in [data_module.train_dataloader(), data_module.val_dataloader()]:

        for batch in dataloader:
            torch.testing.assert_close(
                batch[LATTICE_PARAMETERS], batch[NOISY_LATTICE_PARAMETERS]
            )
            torch.testing.assert_close(
                batch[RELATIVE_COORDINATES], batch[NOISY_RELATIVE_COORDINATES]
            )
            assert not torch.equal(batch[ATOM_TYPES], batch[NOISY_ATOM_TYPES])

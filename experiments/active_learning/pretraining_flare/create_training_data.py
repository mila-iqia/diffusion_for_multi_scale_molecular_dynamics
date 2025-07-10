"""Create training data.

This script trains a FLARE model using Si 2x2x2 data that was generated to train diffusion models.
Only a few batches of that dataset are actually used.
"""
import logging
import pickle
from typing import Dict, List

import lightning as pl

from diffusion_for_multi_scale_molecular_dynamics import DATA_DIR, TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.structure_converter import \
    StructureConverter
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_single_point_calculator import \
    SinglePointCalculation  # noqa
from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import (
    LammpsDataModuleParameters, LammpsForDiffusionDataModule)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters


def get_labelled_structures(batch: Dict, structure_converter: StructureConverter) -> List[SinglePointCalculation]:
    """Get labelled structures.

    A convenience method to extract SinglePointCalculation objects from AXL data.

    Args:
        batch: A batch of data sample.
        structure_converter: A structure converter to go from atomic indices to atom identifiers.

    """
    list_labelled_structures = []
    batch_size = batch['relative_coordinates'].shape[0]
    for idx in range(batch_size):
        forces = batch['cartesian_forces'][idx].numpy()
        axl_structure = AXL(A=batch['atom_types'][idx].numpy(),
                            X=batch['relative_coordinates'][idx].numpy(),
                            L=batch['lattice_parameters'][idx].numpy(),)
        structure = structure_converter.convert_axl_to_structure(axl_structure)

        energy = batch['potential_energy'][idx].item()

        labelled_structure = SinglePointCalculation(calculation_type="stillinger_weber",
                                                    structure=structure,
                                                    forces=forces,
                                                    energy=energy)
        list_labelled_structures.append(labelled_structure)

    return list_labelled_structures


experiment_dir = TOP_DIR / "experiments/active_learning/pretraining_flare/"
sw_coefficients_file_path = DATA_DIR / "stillinger_weber_coefficients/Si.sw"

output_dir = experiment_dir / "data"
output_dir.mkdir(parents=True, exist_ok=True)

seed = 42

element_list = ["Si"]
variance_type = "local"

dataset_name = "Si_diffusion_2x2x2"
max_atom = 64

lammps_run_dir = DATA_DIR / dataset_name
processed_dataset_dir = lammps_run_dir / "processed"
cache_dir = lammps_run_dir / "cache"

batch_size = 128
data_params = LammpsDataModuleParameters(batch_size=batch_size,
                                         max_atom=max_atom,
                                         noise_parameters=NoiseParameters(total_time_steps=1),  # dummy. Not used.
                                         use_optimal_transport=False,
                                         use_fixed_lattice_parameters=True,
                                         elements=element_list)

structure_converter = StructureConverter(element_list)

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    pl.seed_everything(seed)

    logging.info("Loading data")
    datamodule = LammpsForDiffusionDataModule(
        lammps_run_dir=lammps_run_dir,
        processed_dataset_dir=processed_dataset_dir,
        hyper_params=data_params,
        working_cache_dir=cache_dir,
    )
    datamodule.setup()

    train_batch = next(iter(datamodule.train_dataloader()))
    list_train_labelled_structures = get_labelled_structures(train_batch, structure_converter)

    valid_dataloader = datamodule.val_dataloader()

    number_of_validation_batches = len(valid_dataloader)
    list_valid_labelled_structures = None
    list_test_labelled_structures = None
    for batch_idx, batch in enumerate(valid_dataloader):
        if batch_idx == 0:
            list_valid_labelled_structures = get_labelled_structures(batch, structure_converter)
        if batch_idx == number_of_validation_batches - 2:
            list_test_labelled_structures = get_labelled_structures(batch, structure_converter)

    list_file_names = ['train_labelled_structures.pkl',
                       'valid_labelled_structures.pkl',
                       'test_labelled_structures.pkl']

    list_datasets = [list_train_labelled_structures,
                     list_valid_labelled_structures,
                     list_test_labelled_structures]

    for dataset, filename in zip(list_datasets, list_file_names):
        with open(output_dir / filename, "wb") as fd:
            pickle.dump(dataset, fd)

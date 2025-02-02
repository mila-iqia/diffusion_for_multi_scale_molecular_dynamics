from collections import defaultdict
from typing import Dict, List

import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import (
    LammpsDataModuleParameters, LammpsForDiffusionDataModule)
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, CARTESIAN_FORCES, CARTESIAN_POSITIONS, RELATIVE_COORDINATES)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from tests.conftest import TestDiffusionDataBase
from tests.fake_data_utils import Configuration, generate_fake_configuration


def convert_configurations_to_dataset(
    configurations: List[Configuration],
    element_types: ElementTypes,
) -> Dict[str, torch.Tensor]:
    """Convert the input configuration into a dict of torch tensors comparable to a pytorch dataset."""
    # The expected dataset keys are {'natom', 'box', 'cartesian_positions', 'relative_positions', 'type',
    # 'cartesian_forces', 'potential_energy'}
    data = defaultdict(list)
    for configuration in configurations:
        data["natom"].append(len(configuration.ids))
        data["box"].append(configuration.cell_dimensions)
        data[CARTESIAN_FORCES].append(configuration.cartesian_forces)
        data[CARTESIAN_POSITIONS].append(configuration.cartesian_positions)
        data[RELATIVE_COORDINATES].append(configuration.relative_coordinates)
        data[ATOM_TYPES].append([element_types.get_element_id(element) for element in configuration.elements])
        data["potential_energy"].append(configuration.potential_energy)

    configuration_dataset = dict()
    for key, array in data.items():
        configuration_dataset[key] = torch.tensor(array)

    return configuration_dataset


class TestLammpsForDiffusionDataModuleBase(TestDiffusionDataBase):

    @pytest.fixture
    def element_types(self, unique_elements):
        return ElementTypes(unique_elements)

    @pytest.fixture()
    def batch_size(self):
        return 4

    @pytest.fixture
    def batch_of_configurations(self, spatial_dimension, number_of_atoms, unique_elements, batch_size):
        return [generate_fake_configuration(spatial_dimension, number_of_atoms, unique_elements)
                for _ in range(batch_size)]

    @pytest.fixture
    def batched_input_data(self, batch_of_configurations):
        data = defaultdict(list)
        for configuration in batch_of_configurations:
            data["natom"].append(len(configuration.ids))
            data["box"].append(configuration.cell_dimensions.astype(np.float32))
            data[CARTESIAN_FORCES].append(configuration.cartesian_forces.flatten().astype(np.float32))
            data[CARTESIAN_POSITIONS].append(configuration.cartesian_positions.flatten().astype(np.float32))
            data[RELATIVE_COORDINATES].append(configuration.relative_coordinates.flatten().astype(np.float32))
            data['element'].append(configuration.elements)
            data["potential_energy"].append(configuration.potential_energy)

        return data

    @pytest.fixture
    def input_data_for_padding(self, batched_input_data):
        row = dict()
        for key, list_of_values in batched_input_data.items():
            row[key] = list_of_values[0]
        return row

    @pytest.fixture()
    def max_atom_for_padding(self, number_of_atoms):
        return number_of_atoms + 4

    @pytest.fixture
    def data_module_hyperparameters(self, number_of_atoms, spatial_dimension, unique_elements):
        return LammpsDataModuleParameters(
            batch_size=2,
            num_workers=0,
            max_atom=number_of_atoms,
            spatial_dimension=spatial_dimension,
            elements=unique_elements,
            noise_parameters=NoiseParameters(total_time_steps=10)
        )

    @pytest.fixture()
    def data_module(self, paths, data_module_hyperparameters, tmpdir):

        data_module = LammpsForDiffusionDataModule(
            lammps_run_dir=paths["raw_data_dir"],
            processed_dataset_dir=paths["processed_data_dir"],
            hyper_params=data_module_hyperparameters,
            working_cache_dir=tmpdir,
        )
        data_module.setup()

        return data_module

    @pytest.fixture()
    def real_and_test_datasets(
        self, mode, data_module, all_train_configurations, all_valid_configurations, element_types
    ):

        match mode:
            case "train":
                data_module_dataset = data_module.train_dataset[:]
                configuration_dataset = convert_configurations_to_dataset(
                    all_train_configurations, element_types
                )
            case "valid":
                data_module_dataset = data_module.valid_dataset[:]
                configuration_dataset = convert_configurations_to_dataset(
                    all_valid_configurations, element_types
                )
            case _:
                raise ValueError(f"Unknown mode {mode}")

        return data_module_dataset, configuration_dataset

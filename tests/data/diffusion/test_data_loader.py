from collections import defaultdict
from typing import Dict, List

import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import (
    LammpsForDiffusionDataModule, LammpsLoaderParameters)
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import (
    NULL_ELEMENT, ElementTypes)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, CARTESIAN_FORCES, CARTESIAN_POSITIONS, RELATIVE_COORDINATES)
from tests.conftest import TestDiffusionDataBase
from tests.fake_data_utils import (Configuration, find_aligning_permutation,
                                   generate_fake_configuration)


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


class TestDiffusionDataLoader(TestDiffusionDataBase):

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

    def test_dataset_transform(self, batched_input_data, element_types, batch_size, number_of_atoms, spatial_dimension):
        result = LammpsForDiffusionDataModule.dataset_transform(batched_input_data, element_types)
        # Check keys in result
        assert set(result.keys()) == {
            "natom",
            ATOM_TYPES,
            CARTESIAN_FORCES,
            CARTESIAN_POSITIONS,
            RELATIVE_COORDINATES,
            "box",
            "potential_energy",
        }

        # Check tensor types and shapes
        assert torch.equal(
            result["natom"], torch.tensor(batched_input_data["natom"]).long()
        )
        assert result[CARTESIAN_POSITIONS].shape == (
            batch_size,
            number_of_atoms,
            spatial_dimension,
        )
        assert result["box"].shape == (batch_size, spatial_dimension)

        element_ids = list(result[ATOM_TYPES].flatten().numpy())
        computed_element_names = [element_types.get_element(id) for id in element_ids]
        expected_element_names = list(np.array(batched_input_data['element']).flatten())
        assert computed_element_names == expected_element_names

        assert torch.equal(
            result["potential_energy"],
            torch.tensor(batched_input_data["potential_energy"]),
        )

        # Check tensor types explicitly
        assert result["natom"].dtype == torch.long
        assert (
            result[CARTESIAN_POSITIONS].dtype == torch.float32
        )  # default dtype for torch.as_tensor with float inputs
        assert result["box"].dtype == torch.float32
        assert result[ATOM_TYPES].dtype == torch.long
        assert result["potential_energy"].dtype == torch.float32

    @pytest.fixture()
    def max_atom_for_padding(self, number_of_atoms):
        return number_of_atoms + 4

    def test_pad_dataset(self, input_data_for_padding, number_of_atoms, max_atom_for_padding):
        padded_sample = LammpsForDiffusionDataModule.pad_samples(input_data_for_padding, max_atom_for_padding)

        # Check if the type and position have been padded correctly
        assert len(padded_sample["element"]) == max_atom_for_padding
        assert padded_sample[CARTESIAN_POSITIONS].shape == torch.Size([max_atom_for_padding * 3])

        # Check that the padding is correct
        for k in range(number_of_atoms, max_atom_for_padding):
            assert padded_sample["element"][k] == NULL_ELEMENT

        # Check that the padding uses nan for position
        assert torch.isnan(
            padded_sample[CARTESIAN_POSITIONS][3 * number_of_atoms:]
        ).all()

    @pytest.fixture
    def data_module_hyperparameters(self, number_of_atoms, spatial_dimension, unique_elements):
        return LammpsLoaderParameters(
            batch_size=2,
            num_workers=0,
            max_atom=number_of_atoms,
            spatial_dimension=spatial_dimension,
            elements=unique_elements
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

    def test_dataset_feature_names(self, data_module):
        expected_feature_names = {
            "natom",
            "box",
            'element',
            "potential_energy",
            CARTESIAN_FORCES,
            CARTESIAN_POSITIONS,
            RELATIVE_COORDINATES,
        }
        assert set(data_module.train_dataset.features.keys()) == expected_feature_names
        assert set(data_module.valid_dataset.features.keys()) == expected_feature_names

    @pytest.mark.parametrize("mode", ["train", "valid"])
    def test_dataset(self, real_and_test_datasets):

        data_module_dataset, configuration_dataset = real_and_test_datasets

        assert set(data_module_dataset.keys()) == set(configuration_dataset.keys())

        # the configurations and the data module dataset might not be in the same order. Try to build a mapping.
        dataset_boxes = data_module_dataset["box"]
        configuration_boxes = configuration_dataset["box"]

        permutation_indices = find_aligning_permutation(
            dataset_boxes, configuration_boxes
        )

        for field_name in data_module_dataset.keys():
            computed_values = data_module_dataset[field_name]
            expected_values = configuration_dataset[field_name][permutation_indices]

            torch.testing.assert_close(
                computed_values, expected_values, check_dtype=False
            )

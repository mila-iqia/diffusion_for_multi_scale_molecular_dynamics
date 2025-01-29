import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import \
    LammpsForDiffusionDataModule
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    NULL_ELEMENT
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, CARTESIAN_FORCES, CARTESIAN_POSITIONS, RELATIVE_COORDINATES)
from tests.data.diffusion.conftest import TestLammpsForDiffusionDataModuleBase
from tests.fake_data_utils import find_aligning_permutation


class TestLammpsForDiffusionDataModule(TestLammpsForDiffusionDataModuleBase):

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

        assert set(configuration_dataset.keys()).issubset(set(data_module_dataset.keys()))

        # the configurations and the data module dataset might not be in the same order. Try to build a mapping.
        dataset_boxes = data_module_dataset["box"]
        configuration_boxes = configuration_dataset["box"]

        permutation_indices = find_aligning_permutation(
            dataset_boxes, configuration_boxes
        )

        for field_name in configuration_dataset.keys():
            computed_values = data_module_dataset[field_name]
            expected_values = configuration_dataset[field_name][permutation_indices]

            torch.testing.assert_close(
                computed_values, expected_values, check_dtype=False
            )

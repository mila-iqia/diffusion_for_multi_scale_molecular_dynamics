from collections import defaultdict
from typing import Dict, List

import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.data_loader import (
    LammpsForDiffusionDataModule, LammpsLoaderParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    CARTESIAN_FORCES, CARTESIAN_POSITIONS, RELATIVE_COORDINATES)
from tests.conftest import TestDiffusionDataBase
from tests.fake_data_utils import Configuration, find_aligning_permutation


def convert_configurations_to_dataset(
    configurations: List[Configuration],
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
        data["type"].append(configuration.types)
        data["potential_energy"].append(configuration.potential_energy)

    configuration_dataset = dict()
    for key, array in data.items():
        configuration_dataset[key] = torch.tensor(array)

    return configuration_dataset


class TestDiffusionDataLoader(TestDiffusionDataBase):
    @pytest.fixture
    def input_data_to_transform(self):
        return {
            "natom": [2],  # batch size of 1
            "box": [[1.0, 1.0, 1.0]],
            CARTESIAN_POSITIONS: [
                [1.0, 2.0, 3, 4.0, 5, 6]
            ],  # for one batch, two atoms, 3D positions
            CARTESIAN_FORCES: [
                [11.0, 12.0, 13, 14.0, 15, 16]
            ],  # for one batch, two atoms, 3D forces
            RELATIVE_COORDINATES: [[1.0, 2.0, 3, 4.0, 5, 6]],
            "type": [[1, 2]],
            "potential_energy": [23.233],
        }

    def test_dataset_transform(self, input_data_to_transform):
        result = LammpsForDiffusionDataModule.dataset_transform(input_data_to_transform)
        # Check keys in result
        assert set(result.keys()) == {
            "natom",
            CARTESIAN_FORCES,
            CARTESIAN_POSITIONS,
            RELATIVE_COORDINATES,
            "box",
            "type",
            "potential_energy",
        }

        # Check tensor types and shapes
        assert torch.equal(
            result["natom"], torch.tensor(input_data_to_transform["natom"]).long()
        )
        assert result[CARTESIAN_POSITIONS].shape == (
            1,
            2,
            3,
        )  # (batchsize, natom, 3 [since it's 3D])
        assert result["box"].shape == (1, 3)
        assert torch.equal(
            result["type"], torch.tensor(input_data_to_transform["type"]).long()
        )
        assert torch.equal(
            result["potential_energy"],
            torch.tensor(input_data_to_transform["potential_energy"]),
        )

        # Check tensor types explicitly
        assert result["natom"].dtype == torch.long
        assert (
            result[CARTESIAN_POSITIONS].dtype == torch.float32
        )  # default dtype for torch.as_tensor with float inputs
        assert result["box"].dtype == torch.float32
        assert result["type"].dtype == torch.long
        assert result["potential_energy"].dtype == torch.float32

    @pytest.fixture
    def input_data_to_pad(self):
        return {
            "natom": 2,  # batch size of 1
            "box": [1.0, 1.0, 1.0],
            CARTESIAN_POSITIONS: [
                1.0,
                2.0,
                3,
                4.0,
                5,
                6,
            ],  # for one batch, two atoms, 3D positions
            CARTESIAN_FORCES: [11.0, 12.0, 13, 14.0, 15, 16],
            RELATIVE_COORDINATES: [1.0, 2.0, 3, 4.0, 5, 6],
            "type": [1, 2],
            "potential_energy": 23.233,
        }

    def test_pad_dataset(self, input_data_to_pad):
        max_atom = 5  # Assume we want to pad to a max of 5 atoms
        padded_sample = LammpsForDiffusionDataModule.pad_samples(
            input_data_to_pad, max_atom
        )

        # Check if the type and position have been padded correctly
        assert len(padded_sample["type"]) == max_atom
        assert padded_sample[CARTESIAN_POSITIONS].shape == torch.Size([max_atom * 3])

        # Check that the padding uses -1 for type
        # 2 atoms in the input_data - last 3 atoms should be type -1
        for k in range(max_atom - 2):
            assert padded_sample["type"].tolist()[-(k + 1)] == -1

        # Check that the padding uses nan for position
        assert torch.isnan(
            padded_sample[CARTESIAN_POSITIONS][-(max_atom - 2) * 3:]
        ).all()

    @pytest.fixture
    def data_module_hyperparameters(self, number_of_atoms, spatial_dimension):
        return LammpsLoaderParameters(
            batch_size=2,
            num_workers=0,
            max_atom=number_of_atoms,
            spatial_dimension=spatial_dimension,
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
        self, mode, data_module, all_train_configurations, all_valid_configurations
    ):

        match mode:
            case "train":
                data_module_dataset = data_module.train_dataset[:]
                configuration_dataset = convert_configurations_to_dataset(
                    all_train_configurations
                )
            case "valid":
                data_module_dataset = data_module.valid_dataset[:]
                configuration_dataset = convert_configurations_to_dataset(
                    all_valid_configurations
                )
            case _:
                raise ValueError(f"Unknown mode {mode}")

        return data_module_dataset, configuration_dataset

    def test_dataset_feature_names(self, data_module):
        expected_feature_names = {
            "natom",
            "box",
            "type",
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

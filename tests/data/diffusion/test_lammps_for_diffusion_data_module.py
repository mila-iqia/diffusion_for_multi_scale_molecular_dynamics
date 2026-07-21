import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import (
    LammpsDataModuleParameters, LammpsForDiffusionDataModule)
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    NULL_ELEMENT
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, CARTESIAN_FORCES, CARTESIAN_POSITIONS, LATTICE_PARAMETERS,
    NOISY_ATOM_TYPES, NOISY_RELATIVE_COORDINATES, NUMBER_OF_ATOMS,
    PADDED_ATOM_TYPE, RELATIVE_COORDINATES)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from tests.data.diffusion.conftest import TestLammpsForDiffusionDataModuleBase
from tests.fake_data_utils import (create_dump_yaml_documents,
                                   create_thermo_yaml_documents,
                                   find_aligning_permutation,
                                   generate_fake_configuration, write_to_yaml)


class TestLammpsForDiffusionDataModule(TestLammpsForDiffusionDataModuleBase):

    def test_dataset_transform(
        self,
        batched_input_data,
        element_types,
        batch_size,
        number_of_atoms,
        spatial_dimension,
    ):
        result = LammpsForDiffusionDataModule.dataset_transform(
            batched_input_data, element_types
        )
        # Check keys in result
        assert set(result.keys()) == {
            "natom",
            ATOM_TYPES,
            CARTESIAN_FORCES,
            CARTESIAN_POSITIONS,
            RELATIVE_COORDINATES,
            LATTICE_PARAMETERS,
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
        assert result[LATTICE_PARAMETERS].shape == (
            batch_size,
            int(spatial_dimension * (spatial_dimension + 1) / 2),
        )

        element_ids = list(result[ATOM_TYPES].flatten().numpy())
        computed_element_names = [element_types.get_element(id) for id in element_ids]
        expected_element_names = list(np.array(batched_input_data["element"]).flatten())
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

    def test_pad_dataset(
        self, input_data_for_padding, number_of_atoms, max_atom_for_padding
    ):
        padded_sample = LammpsForDiffusionDataModule.pad_samples(
            input_data_for_padding, max_atom_for_padding
        )

        # Check if the type and position have been padded correctly
        assert len(padded_sample["element"]) == max_atom_for_padding
        assert padded_sample[CARTESIAN_POSITIONS].shape == torch.Size(
            [max_atom_for_padding * 3]
        )

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
            "element",
            "potential_energy",
            CARTESIAN_FORCES,
            CARTESIAN_POSITIONS,
            RELATIVE_COORDINATES,
            LATTICE_PARAMETERS,
        }
        assert set(data_module.train_dataset.features.keys()) == expected_feature_names
        assert set(data_module.valid_dataset.features.keys()) == expected_feature_names

    @pytest.mark.parametrize("mode", ["train", "valid"])
    def test_dataset(self, real_and_test_datasets):

        data_module_dataset, configuration_dataset = real_and_test_datasets

        assert set(configuration_dataset.keys()).issubset(
            set(data_module_dataset.keys())
        )

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


class TestLammpsForDiffusionDataModulePadding:

    @pytest.fixture()
    def spatial_dimension(self):
        return 3

    @pytest.fixture()
    def unique_elements(self):
        return ["Si"]

    @pytest.fixture()
    def natoms_per_run(self):
        return [3, 5, 4]

    @pytest.fixture()
    def paths(self, tmp_path, natoms_per_run, spatial_dimension, unique_elements):
        raw_data_dir = tmp_path / "raw_data"
        raw_data_dir.mkdir()

        for mode in ["train", "valid"]:
            for i, n in enumerate(natoms_per_run, 1):
                run_directory = raw_data_dir / f"{mode}_run_{i}"
                run_directory.mkdir()
                configurations = [
                    generate_fake_configuration(spatial_dimension, n, unique_elements)
                    for _ in range(2)
                ]
                dump_docs = create_dump_yaml_documents(configurations)
                thermo_docs = create_thermo_yaml_documents(configurations)
                write_to_yaml(dump_docs, str(run_directory / f"dump_{mode}.yaml"))
                write_to_yaml(thermo_docs, str(run_directory / "thermo_logs.yaml"))

        processed_data_dir = tmp_path / "processed_data"
        processed_data_dir.mkdir()

        return dict(
            raw_data_dir=str(raw_data_dir), processed_data_dir=str(processed_data_dir)
        )

    @pytest.fixture()
    def data_module(self, paths, natoms_per_run, unique_elements, spatial_dimension, tmp_path):
        hyper_params = LammpsDataModuleParameters(
            elements=unique_elements,
            batch_size=16,
            num_workers=0,
            max_atom=max(natoms_per_run),
            spatial_dimension=spatial_dimension,
            noise_parameters=NoiseParameters(total_time_steps=10),
            use_fixed_lattice_parameters=True,
        )
        dm = LammpsForDiffusionDataModule(
            lammps_run_dir=paths["raw_data_dir"],
            processed_dataset_dir=paths["processed_data_dir"],
            hyper_params=hyper_params,
            working_cache_dir=str(tmp_path / "cache"),
        )
        dm.setup()
        return dm

    def test_padding(self, data_module, natoms_per_run):
        dataset = data_module.train_dataset[:]
        noisy_atom_types = dataset[NOISY_ATOM_TYPES]        # [n_structures, max_atom]
        noisy_coords = dataset[NOISY_RELATIVE_COORDINATES]  # [n_structures, max_atom, 3]
        natoms = dataset[NUMBER_OF_ATOMS]                   # [n_structures]

        max_atom = max(natoms_per_run)

        for i in range(len(natoms)):
            n = int(natoms[i])
            if n < max_atom:
                assert (noisy_atom_types[i, n:] == PADDED_ATOM_TYPE).all(), (
                    f"Sample {i} with {n} atoms: padded atom types at positions [{n}:] should be {PADDED_ATOM_TYPE}"
                )
                assert torch.isnan(noisy_coords[i, n:, :]).all(), (
                    f"Sample {i} with {n} atoms: padded coordinates at positions [{n}:] should be NaN"
                )

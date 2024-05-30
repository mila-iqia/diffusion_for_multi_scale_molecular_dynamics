import os

import numpy as np
import pandas as pd
import pytest

from crystal_diffusion.data.diffusion.data_preprocess import \
    LammpsProcessorForDiffusion
from crystal_diffusion.namespace import CARTESIAN_POSITIONS, CARTESIAN_FORCES, RELATIVE_COORDINATES
from tests.conftest import TestDiffusionDataBase
from tests.fake_data_utils import generate_parquet_dataframe


class TestDataProcess(TestDiffusionDataBase):

    @pytest.fixture
    def processor(self, paths):
        return LammpsProcessorForDiffusion(**paths)

    def test_prepare_train_data(self, processor, paths, train_configuration_runs, number_of_train_runs):
        list_files = processor.prepare_data(paths['raw_data_dir'], mode='train')
        assert len(list_files) == number_of_train_runs

        for run_number, configurations in enumerate(train_configuration_runs, 1):
            expected_parquet_file = os.path.join(processor.data_dir, f"train_run_{run_number}.parquet")
            assert expected_parquet_file in list_files

            computed_df = pd.read_parquet(expected_parquet_file)
            expected_df = generate_parquet_dataframe(configurations)
            pd.testing.assert_frame_equal(computed_df, expected_df)

    def test_prepare_valid_data(self, processor, paths, valid_configuration_runs, number_of_valid_runs):
        list_files = processor.prepare_data(paths['raw_data_dir'], mode='valid')
        assert len(list_files) == number_of_valid_runs

        for run_number, configurations in enumerate(valid_configuration_runs, 1):
            expected_parquet_file = os.path.join(processor.data_dir, f"valid_run_{run_number}.parquet")
            assert expected_parquet_file in list_files

            computed_df = pd.read_parquet(expected_parquet_file)
            expected_df = generate_parquet_dataframe(configurations)
            pd.testing.assert_frame_equal(computed_df, expected_df)

    def test_parse_lammps_run(self, processor, paths, train_configuration_runs, valid_configuration_runs):
        expected_columns = ['natom', 'box', 'type', CARTESIAN_POSITIONS, CARTESIAN_FORCES, RELATIVE_COORDINATES,
                            'potential_energy']

        for mode, configuration_runs in zip(['train', 'valid'], [train_configuration_runs, valid_configuration_runs]):

            for run_number, configurations in enumerate(configuration_runs, 1):
                run_dir = os.path.join(paths['raw_data_dir'], f"{mode}_run_{run_number}")
                computed_df = processor.parse_lammps_run(run_dir)
                assert computed_df is not None
                assert not computed_df.empty
                for column_name in expected_columns:
                    assert column_name in computed_df.columns

                expected_df = generate_parquet_dataframe(configurations)
                pd.testing.assert_frame_equal(computed_df, expected_df)

    @pytest.fixture
    def box_coordinates(self):
        return [1, 2, 3]

    @pytest.fixture
    def sample_coordinates(self, box_coordinates):
        # Sample data frame
        return pd.DataFrame({
            'box': [box_coordinates],
            'x': [[0.6, 0.06, 0.006, 0.00006]],
            'y': [[1.2, 0.12, 0.0012, 0.00012]],
            'z': [[1.8, 0.18, 0.018, 0.0018]]
        })

    def test_convert_coords_to_relative(self, sample_coordinates, box_coordinates):
        # Expected output: Each coordinate divided by 1, 2, 3 (the box limits)
        for index, row in sample_coordinates.iterrows():
            relative_coords = LammpsProcessorForDiffusion._convert_coords_to_relative(row)
            expected_coords = []
            for x, y, z in zip(row['x'], row['y'], row['z']):
                expected_coords.extend([x / box_coordinates[0], y / box_coordinates[1], z / box_coordinates[2]])
            assert relative_coords == expected_coords

    def test_convert_coords_to_relative2(self, processor, all_configurations):

        for configuration in all_configurations:

            natom = len(configuration.ids)
            expected_coordinates = configuration.relative_coordinates
            positions = configuration.cartesian_positions
            box = configuration.cell_dimensions

            position_series = pd.Series({'x': positions[:, 0], 'y': positions[:, 1], 'z': positions[:, 2], 'box': box})

            computed_coordinates = np.array(processor._convert_coords_to_relative(position_series)).reshape(natom, 3)
            np.testing.assert_almost_equal(computed_coordinates, expected_coordinates)

    def test_get_x_relative(self, processor, sample_coordinates):
        # Call get_x_relative on the test data
        result_df = processor.get_x_relative(sample_coordinates)
        # Check if 'relative_positions' column is added
        assert RELATIVE_COORDINATES in result_df.columns

    def test_flatten_positions_in_row(self):

        number_of_atoms = 12
        spatial_dimensions = 3
        position_data = np.random.rand(number_of_atoms, spatial_dimensions)
        row = pd.Series(dict(x=list(position_data[:, 0]), y=list(position_data[:, 1]), z=list(position_data[:, 2])))

        computed_flattened_positions = LammpsProcessorForDiffusion._flatten_positions_in_row(row)
        expected_flattened_positions = position_data.flatten()

        np.testing.assert_almost_equal(expected_flattened_positions, computed_flattened_positions)

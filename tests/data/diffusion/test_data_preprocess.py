import os

import numpy as np
import pandas as pd
import pytest

from crystal_diffusion.data.diffusion.data_preprocess import \
    LammpsProcessorForDiffusion
from tests.fake_data_utils import (create_dump_yaml_documents,
                                   create_thermo_yaml_documents,
                                   generate_fake_configuration,
                                   generate_parquet_dataframe, write_to_yaml)


@pytest.fixture
def train_configurations():
    np.random.seed(23423)
    return [generate_fake_configuration(spatial_dimension=3, number_of_atoms=8) for _ in range(16)]


@pytest.fixture
def valid_configurations():
    np.random.seed(1234)
    return [generate_fake_configuration(spatial_dimension=3, number_of_atoms=8) for _ in range(16)]


@pytest.fixture
def configurations(train_configurations, valid_configurations):
    return train_configurations + valid_configurations


@pytest.fixture
def paths(tmp_path, train_configurations, valid_configurations):
    raw_data_dir = tmp_path / "raw_data"
    raw_data_dir.mkdir()

    for mode, configurations in zip(['train', 'valid'], [train_configurations, valid_configurations]):
        run_directory = raw_data_dir / f'{mode}_run_1'
        run_directory.mkdir()
        dump_docs = create_dump_yaml_documents(configurations)
        thermo_docs = create_thermo_yaml_documents(configurations)

        write_to_yaml(dump_docs, str(run_directory / f'dump_{mode}.yaml'))
        write_to_yaml(thermo_docs, str(run_directory / 'thermo_logs.yaml'))

    processed_data_dir = tmp_path / "processed_data"
    processed_data_dir.mkdir()

    return dict(raw_data_dir=str(raw_data_dir), processed_data_dir=str(processed_data_dir))


@pytest.fixture
def processor(paths):
    return LammpsProcessorForDiffusion(**paths)


def test_prepare_data(processor, paths, train_configurations, valid_configurations):
    # Assuming that the raw_data directory is properly set up with train_run_1 and valid_run_1 subdirectory
    for mode, configurations in zip(['train', 'valid'], [train_configurations, valid_configurations]):
        list_files = processor.prepare_data(paths['raw_data_dir'], mode=mode)
        assert len(list_files) == 1

        expected_parquet_file = os.path.join(processor.data_dir, f"{mode}_run_1.parquet")
        assert expected_parquet_file in list_files

        computed_df = pd.read_parquet(expected_parquet_file)
        expected_df = generate_parquet_dataframe(configurations)
        pd.testing.assert_frame_equal(computed_df, expected_df)


def test_parse_lammps_run(processor, paths, train_configurations, valid_configurations):
    # Assuming that the raw_data directory is properly set up with train_run_1 and valid_run_1 subdirectory
    for mode, configurations in zip(['train', 'valid'], [train_configurations, valid_configurations]):
        run_dir = os.path.join(paths['raw_data_dir'], f"{mode}_run_1")
        computed_df = processor.parse_lammps_run(run_dir)
        assert computed_df is not None
        assert not computed_df.empty
        expected_columns = ['natom', 'box', 'type', 'position', 'relative_positions', 'energy']
        for column_name in expected_columns:
            assert column_name in computed_df.columns

        expected_df = generate_parquet_dataframe(configurations)
        pd.testing.assert_frame_equal(computed_df, expected_df)


@pytest.fixture
def box_coordinates():
    return [1, 2, 3]


@pytest.fixture
def sample_coordinates(box_coordinates):
    # Sample data frame
    return pd.DataFrame({
        'box': [box_coordinates],
        'x': [[0.6, 0.06, 0.006, 0.00006]],
        'y': [[1.2, 0.12, 0.0012, 0.00012]],
        'z': [[1.8, 0.18, 0.018, 0.0018]]
    })


def test_convert_coords_to_relative(sample_coordinates, box_coordinates):
    # Expected output: Each coordinate divided by 1, 2, 3 (the box limits)
    for index, row in sample_coordinates.iterrows():
        relative_coords = LammpsProcessorForDiffusion._convert_coords_to_relative(row)
        expected_coords = []
        for x, y, z in zip(row['x'], row['y'], row['z']):
            expected_coords.extend([x / box_coordinates[0], y / box_coordinates[1], z / box_coordinates[2]])
        assert relative_coords == expected_coords


def test_convert_coords_to_relative2(processor, configurations):

    for configuration in configurations:

        natom = len(configuration.ids)
        expected_coordinates = configuration.relative_coordinates
        positions = configuration.positions
        box = configuration.cell_dimensions

        position_series = pd.Series({'x': positions[:, 0], 'y': positions[:, 1], 'z': positions[:, 2], 'box': box})

        computed_coordinates = np.array(processor._convert_coords_to_relative(position_series)).reshape(natom, 3)
        np.testing.assert_almost_equal(computed_coordinates, expected_coordinates)


def test_get_x_relative(processor, sample_coordinates):
    # Call get_x_relative on the test data
    result_df = processor.get_x_relative(sample_coordinates)
    # Check if 'relative_positions' column is added
    assert 'relative_positions' in result_df.columns

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from crystal_diffusion.data.diffusion.data_preprocess import \
    LammpsProcessorForDiffusion


@pytest.fixture
def mock_processor(tmp_path):
    raw_data_dir = tmp_path / "raw_data"
    processed_data_dir = tmp_path / "processed_data"
    raw_data_dir.mkdir()
    processed_data_dir.mkdir()
    processor = LammpsProcessorForDiffusion(str(raw_data_dir), str(processed_data_dir))
    return processor


@pytest.fixture
def mock_parse_lammps_method(monkeypatch):
    def mock_return(*args, **kwargs):
        # Mock DataFrame similar to the expected one from parse_lammps_run
        return pd.DataFrame({
            'id': [1, 2],
            'type': [1, 2],
            'x': [0.1, 0.2],
            'y': [0.2, 0.3],
            'z': [0.3, 0.4],
            'fx': [0.1, -0.1],
            'energy': [1.5, 2.5]
        })

    monkeypatch.setattr(
        'crystal_diffusion.data.diffusion.data_preprocess.LammpsProcessorForDiffusion.parse_lammps_run',
        mock_return
    )


def test_prepare_data(mock_processor, mock_parse_lammps_method, tmp_path):
    # Assuming that the raw_data directory is properly set up with train_run_1 subdirectory
    train_run_dir = os.path.join(tmp_path, "raw_data", "train_run_1")
    os.makedirs(train_run_dir)
    expected_parquet_file = os.path.join(mock_processor.data_dir, "train_run_1.parquet")
    train_files = mock_processor.prepare_data(os.path.join(tmp_path, "raw_data"), mode='train')
    assert len(train_files) == 1
    assert expected_parquet_file in train_files


@pytest.fixture
def mock_parse_lammps_output(monkeypatch):
    # Create a fake parse_lammps_output function
    def mock_parse_lammps_output(*args, **kwargs):
        # Return a fixed DataFrame that imitates the actual parse_lammps_output output
        return pd.DataFrame({
            'id': [[1, 2]],
            'type': [[1, 2]],
            'x': [[0.1, 0.2]],
            'y': [[0.2, 0.3]],
            'z': [[0.3, 0.4]],
            'fx': [[0.1, -0.1]],
            'energy': [[1.5, 2.5]],
            'box': [[1.6, 2.6, 3.6]]
        })
    # Use monkeypatch to replace the actual function with the fake one for tests
    monkeypatch.setattr(
        'crystal_diffusion.data.diffusion.data_preprocess.parse_lammps_output',
        mock_parse_lammps_output
    )


def test_parse_lammps_run(mock_processor, mock_parse_lammps_output, tmp_path):
    # Assuming that the raw_data directory is properly set up with train_run_1 subdirectory
    train_run_dir = os.path.join(tmp_path, "raw_data", "train_run_1")
    os.makedirs(train_run_dir)
    dump_file = os.path.join(train_run_dir, "dump_file")
    thermo_file = os.path.join(train_run_dir, "thermo_file")
    Path(dump_file).touch()  # Create the file
    Path(thermo_file).touch()  # Create the file
    df = mock_processor.parse_lammps_run(train_run_dir)
    assert df is not None
    assert not df.empty
    assert 'natom' in df.columns
    assert 'box' in df.columns
    assert 'type' in df.columns
    assert 'position' in df.columns
    assert 'relative_positions' in df.columns


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


@pytest.fixture
def mock_prepare_data():
    with (patch('crystal_diffusion.data.diffusion.data_preprocess.LammpsProcessorForDiffusion.prepare_data')
          as mock_prepare):
        mock_prepare.return_value = MagicMock()
        yield mock_prepare


def test_get_x_relative(mock_prepare_data, sample_coordinates, tmpdir):
    # Call get_x_relative on the test data
    lp = LammpsProcessorForDiffusion(tmpdir, tmpdir)
    result_df = lp.get_x_relative(sample_coordinates)
    # Check if 'relative_positions' column is added
    assert 'relative_positions' in result_df.columns

import numpy as np
import pytest

from tests.fake_data_utils import (create_dump_yaml_documents,
                                   create_thermo_yaml_documents,
                                   get_configuration_runs, write_to_yaml)


class TestDiffusionDataBase:
    """This base class creates fake data and writes it to disk for testing."""
    @pytest.fixture(scope="class", autouse=True)
    def set_seed(self):
        """Set the random seed."""
        np.random.seed(23423423)

    @pytest.fixture
    def number_of_train_runs(self):
        """Number of train runs."""
        return 3

    @pytest.fixture
    def number_of_valid_runs(self):
        """Number of valid runs."""
        return 2

    @pytest.fixture()
    def number_of_atoms(self):
        """Number of atoms in fake data."""
        return 8

    @pytest.fixture()
    def spatial_dimension(self):
        """Spatial dimension of fake data."""
        return 3

    @pytest.fixture
    def train_configuration_runs(self, number_of_train_runs, spatial_dimension, number_of_atoms):
        """Generate multiple fake 'data' runs and return their configurations."""
        return get_configuration_runs(number_of_train_runs, spatial_dimension, number_of_atoms)

    @pytest.fixture
    def all_train_configurations(self, train_configuration_runs):
        """Combine all training configurations."""
        all_configurations = []
        for list_configs in train_configuration_runs:
            all_configurations += list_configs
        return all_configurations

    @pytest.fixture
    def valid_configuration_runs(self, number_of_valid_runs, spatial_dimension, number_of_atoms):
        """Generate multiple fake 'data' runs and return their configurations."""
        return get_configuration_runs(number_of_valid_runs, spatial_dimension, number_of_atoms)

    @pytest.fixture
    def all_valid_configurations(self, valid_configuration_runs):
        """Combine all validation configurations."""
        all_configurations = []
        for list_configs in valid_configuration_runs:
            all_configurations += list_configs
        return all_configurations

    @pytest.fixture
    def all_configurations(self, all_train_configurations, all_valid_configurations):
        """Combine all configurations."""
        return all_train_configurations + all_valid_configurations

    @pytest.fixture
    def paths(self, tmp_path, train_configuration_runs, valid_configuration_runs):
        """Write to disk all the fake data in the correct format and return the relevant paths."""
        raw_data_dir = tmp_path / "raw_data"
        raw_data_dir.mkdir()

        for mode, list_configurations in zip(['train', 'valid'], [train_configuration_runs, valid_configuration_runs]):
            for i, configurations in enumerate(list_configurations, 1):
                run_directory = raw_data_dir / f'{mode}_run_{i}'
                run_directory.mkdir()
                dump_docs = create_dump_yaml_documents(configurations)
                thermo_docs = create_thermo_yaml_documents(configurations)

                write_to_yaml(dump_docs, str(run_directory / f'dump_{mode}.yaml'))
                write_to_yaml(thermo_docs, str(run_directory / 'thermo_logs.yaml'))

        processed_data_dir = tmp_path / "processed_data"
        processed_data_dir.mkdir()

        return dict(raw_data_dir=str(raw_data_dir), processed_data_dir=str(processed_data_dir))
import os
from unittest.mock import MagicMock, mock_open

import pandas as pd
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.benchmark import \
    ActiveLearningLoop


class TestActiveLearningLoop:
    @pytest.fixture
    def mock_yaml_config(self):
        return """
        active_learning_data:
            key1: value1
        mlip:
            key2: value2
        structure_evaluation:
            key3: value3
        repainting_model:
            key4: value4
        oracle:
            key5: value5
        """

    @pytest.fixture
    def meta_config(self):  # mock a path to a meta_config yaml file
        return "fake_config.yaml"

    @pytest.fixture
    def mock_al_loop(self, mocker, mock_yaml_config, meta_config):
        # Mock the open function to simulate reading the YAML file
        mocker.patch("builtins.open", mock_open(read_data=mock_yaml_config))
        # Mock os.path.exists to always return True
        mocker.patch("os.path.exists", return_value=True)
        # Mock the instantiate function from hydra.utils
        mock_instantiate = mocker.patch(
            "diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.benchmark.instantiate"
        )
        mock_instantiate.side_effect = (
            lambda x: x
        )  # Return the config itself for simplicity

        # Create an instance of ActiveLearningLoop
        loop = ActiveLearningLoop(meta_config)
        return loop

    def test_parse_config(self, mock_al_loop, mock_yaml_config, meta_config):

        # Assertions to verify that the attributes were correctly set
        assert mock_al_loop.data_paths == {"key1": "value1"}
        assert mock_al_loop.mlip_model == {"key2": "value2"}
        assert mock_al_loop.eval_config == {"key3": "value3"}
        assert mock_al_loop.structure_generation == {"key4": "value4"}
        assert mock_al_loop.oracle == {"key5": "value5"}

        # Verify that the file was opened and the path was checked
        open.assert_called_once_with(meta_config, "r")
        os.path.exists.assert_called_once_with(meta_config)

    def test_train_mlip(self, mocker, mock_yaml_config, mock_al_loop):
        # Mocking the mlip_model's methods
        mock_mlip_model = MagicMock()
        mock_mlip_model.prepare_dataset_from_lammps.return_value = "mock_training_set"
        mock_mlip_model.train.return_value = "mock_trained_mlip_model"
        mock_mlip_model.merge_inputs.return_value = "mock_training_set"

        # Inject the mocked mlip_model into the loop instance
        mock_al_loop.mlip_model = mock_mlip_model
        mock_al_loop.data_paths = MagicMock(training_data_dir="mock_training_data_dir")

        # Run the train_mlip method without providing a training_set
        result = mock_al_loop.train_mlip(round=1)

        # Verify the methods were called with expected parameters
        mock_mlip_model.prepare_dataset_from_lammps.assert_called_once_with(
            root_data_dir="mock_training_data_dir",
            atom_dict=mock_al_loop.atom_dict,
            mode="train",
        )

        mock_mlip_model.train.assert_called_once_with(
            "mock_training_set", mlip_name="mlip_round_1"
        )

        # Verify the trained model path is correctly returned
        assert result == "mock_trained_mlip_model"

        # Verify that the trained model is appended to the history
        assert mock_al_loop.trained_mlips == ["mock_trained_mlip_model"]

        # Test when a training set is provided
        custom_training_set = "custom_training_set"
        result = mock_al_loop.train_mlip(round=2, training_set=custom_training_set)

        # The prepare_dataset_from_lammps should not be called since we provided a training_set
        mock_mlip_model.prepare_dataset_from_lammps.assert_called_once()  # No new call
        mock_mlip_model.train.assert_called_with(
            custom_training_set, mlip_name="mlip_round_2"
        )

        assert result == "mock_trained_mlip_model"
        assert mock_al_loop.trained_mlips == [
            "mock_trained_mlip_model",
            "mock_trained_mlip_model",
        ]

    def test_evaluate_mlip(self, mock_al_loop, tmpdir):
        # Mocking the mlip_model's methods
        mock_mlip_model = MagicMock()
        mock_evaluation_dataset = "mock_evaluation_dataset"
        mock_prediction_df = pd.DataFrame({"atom_index": [0, 1], "force": [1.0, 2.0]})

        # Mocking return values for the prepare_dataset_from_lammps and evaluate methods
        mock_mlip_model.prepare_dataset_from_lammps.return_value = (
            mock_evaluation_dataset
        )
        mock_mlip_model.evaluate.return_value = (None, mock_prediction_df)

        loop = mock_al_loop

        # Inject the mocked mlip_model into the loop instance
        loop.mlip_model = mock_mlip_model
        loop.data_paths = MagicMock(evaluation_data_dir="mock_evaluation_data_dir")

        # Run the evaluate_mlip method without specifying mlip_name
        result_df = loop.evaluate_mlip(round=1)

        # Verify the prepare_dataset_from_lammps method was called with expected parameters
        mock_mlip_model.prepare_dataset_from_lammps.assert_called_once_with(
            root_data_dir="mock_evaluation_data_dir",
            atom_dict=loop.atom_dict,
            mode="evaluation",
            get_forces=True,
        )
        # Verify the evaluate method was called with the correct parameters
        expected_mlip_name = os.path.join(mock_mlip_model.savedir, "mlip_round_1.almtp")
        mock_mlip_model.evaluate.assert_called_once_with(
            mock_evaluation_dataset, mlip_name=expected_mlip_name
        )

        # Verify the method returns the correct dataframe
        pd.testing.assert_frame_equal(result_df, mock_prediction_df)

        # Run the evaluate_mlip method with a custom mlip_name
        custom_mlip_name = "custom_mlip.almtp"
        result_df = loop.evaluate_mlip(round=2, mlip_name=custom_mlip_name)

        # The evaluate method should be called with the custom mlip_name
        mock_mlip_model.evaluate.assert_called_with(
            mock_evaluation_dataset, mlip_name=custom_mlip_name
        )

        pd.testing.assert_frame_equal(result_df, mock_prediction_df)

        # Test without forces_available
        _ = loop.evaluate_mlip(round=3, forces_available=False)

        mock_mlip_model.prepare_dataset_from_lammps.assert_called_with(
            root_data_dir="mock_evaluation_data_dir",
            atom_dict=loop.atom_dict,
            mode="evaluation",
            get_forces=False,
        )

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Structure

from crystal_diffusion.models.mtp import MTPWithMLIP3


class FakeStructure:
    """Mock a pymatgen structure"""
    def __init__(self, species):
        self.species = species


def passthrough(*args, **kwargs):
    """Return arguments as passed.

    Useful for mocking a function to return its input arguments directly.
    """
    return args if len(kwargs) == 0 else (args, kwargs)


@pytest.fixture
def mock_popen(mocker):
    # mock subprocess calling mlp
    mock_popen = mocker.patch("subprocess.Popen")
    mock_popen.return_value.__enter__.return_value.communicate.return_value = (b'', b'')  # stdout, stderr
    mock_popen.return_value.__enter__.return_value.returncode = 0
    return mock_popen


# Mock the external dependencies and method calls within the MTPWithMLIP3.train method
def test_train(mocker, mock_popen):
    # Mock os.path.exists to always return True
    mocker.patch("os.path.exists", return_value=True)

    # Mock check_structures_forces_stresses to return a value without needing real input
    mocker.patch("crystal_diffusion.models.mtp.check_structures_forces_stresses", side_effect=passthrough)

    # Mock pool_from to return a simplified pool object
    mocker.patch("crystal_diffusion.models.mtp.pool_from", return_value="simple_pool_object")

    # Mock self.write_cfg to simulate creating a config file without file operations
    mocker.patch.object(MTPWithMLIP3, "write_cfg", return_value="mock_filename.cfg")

    mocker.patch("shutil.copyfile", return_value=None)

    # Initialize MTPWithMLIP3 with mock parameters
    model = MTPWithMLIP3(mlip_path="/mock/path", name="test_model")
    # Call the train method

    return_code = model.train(
        train_structures=[FakeStructure(['H', 'O']), FakeStructure(['Si'])],
        train_energies=[1, 2],
        train_forces=[],
        train_stresses=[],
        unfitted_mtp="08.almtp",
        fitted_mtp_savedir="/mock/dir"
    )
    # Assert the expected results
    assert return_code == 0  # The train method should return the mocked subprocess success return code from mock_open

    # Assert that mocked methods were called
    model.write_cfg.assert_called()


@pytest.fixture
def fake_structure():
    # Setup a simple mock structure object
    # Replace with appropriate structure setup for your use case
    return Structure(lattice=[1, 0, 0, 0, 1, 0, 0, 0, 1], species=[""], coords=[[0, 0, 0]])


@pytest.fixture
def mtp_instance(mocker):
    # Mock __init__ to not execute its original behavior
    mocker.patch.object(MTPWithMLIP3, '__init__', lambda x, y: None)
    # Setup a mocked instance with necessary attributes
    instance = MTPWithMLIP3("mock_path")
    instance.mlp_command = "mock_mlp_command"
    instance.fitted_mtp = "mock_fitted_mtp"
    instance.elements = ["Si"]
    return instance


# def test_evaluate(mocker, fake_structure, mtp_instance, mock_popen):
def test_evaluate(mocker, fake_structure, mtp_instance, mock_popen):
    test_structures = [fake_structure]
    test_energies = [1.0]
    test_forces = [[[0, 0, 0]]]
    test_stresses = None  # or appropriate mock stresses

    # Mock check_structures_forces_stresses to return the arguments unmodified
    mocker.patch("crystal_diffusion.models.mtp.check_structures_forces_stresses",
                 side_effect=lambda s, f, st: (s, f, st))

    # Mock pool_from to return a mocked value
    mocker.patch("crystal_diffusion.models.mtp.pool_from", return_value="mock_pool")

    # Mock self.write_cfg to simulate creating a config file without file operations
    mocker.patch.object(MTPWithMLIP3, "write_cfg", return_value="mock_filename.cfg")

    # Mock read_cfgs to simulate reading of configurations without accessing the file system
    mocker.patch.object(MTPWithMLIP3, "read_cfgs", return_value="mock_dataframe")

    # Mock os.remove, shutil.copyfile and os.path.exists since evaluate interacts with the filesystem
    mocker.patch("os.remove", return_value=None)
    mocker.patch("shutil.copyfile", return_value=None)
    mocker.patch("os.path.exists", return_value=True)

    # Perform the test
    df_orig, df_predict = mtp_instance.evaluate(test_structures, test_energies, test_forces, test_stresses)

    # Assertions can vary based on the real output of `read_cfgs`
    # Here's an example assertion assuming `read_cfgs` returns a string in this mocked scenario
    assert df_orig == "mock_dataframe" and df_predict == "mock_dataframe", "Evaluate method should return mock" + \
                                                                           "dataframes"


def test_read_cfgs(mtp_instance):
    cfg_path = Path(__file__).parent.joinpath("mtp_cfg_examples.txt")
    df = mtp_instance.read_cfgs(cfg_path, True)
    print(df.keys())
    assert np.array_equal(df['x'], [0.1, 0.2, 0.3])
    assert np.array_equal(df['y'], [1.1, 1.2, 1.3])
    assert np.array_equal(df['z'], [2.1, 2.2, 2.3])
    assert np.array_equal(df['fx'], [3.1, 3.2, 3.3])
    assert np.array_equal(df['fy'], [4.1, 4.2, 4.3])
    assert np.array_equal(df['fz'], [5.1, 5.2, 5.3])
    assert np.array_equal(df['nbh_grades'], [6.1, 6.2, 6.3])

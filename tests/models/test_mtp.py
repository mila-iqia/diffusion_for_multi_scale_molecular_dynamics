import pytest

from crystal_diffusion.models.mtp import MTPWithMLIP3


class MockStructure:
    """Mock a pymatgen structure"""
    def __init__(self, species):
        self.species = species

def passthrough(*args, **kwargs):
    """Function to return arguments as passed.

       Useful for mocking a function to return its input arguments directly.
    """
    return args if len(kwargs) == 0 else (args, kwargs)


# Mock the external dependencies and method calls within the MTPWithMLIP3.train method
@pytest.mark.parametrize("mock_subprocess", [0])  # Here, 0 simulates a successful subprocess return code
def test_train_method(mocker, mock_subprocess):
    # Mock os.path.exists to always return True
    mocker.patch("os.path.exists", return_value=True)

    # Mock the ScratchDir context manager to do nothing
    mocker.patch("crystal_diffusion.models.mtp.ScratchDir", mocker.MagicMock())

    # Mock check_structures_forces_stresses to return a value without needing real input
    mocker.patch("crystal_diffusion.models.mtp.check_structures_forces_stresses", side_effect=passthrough)

    # Mock pool_from to return a simplified pool object
    mocker.patch("crystal_diffusion.models.mtp.pool_from", return_value="simple_pool_object")

    # Mock self.write_cfg to simulate creating a config file without file operations
    mocker.patch.object(MTPWithMLIP3, "write_cfg", return_value="mock_filename.cfg")

    # mocker.patch("crystal_diffusion.models.mtp.itertools.chain", return_value=[1, 2, 3])
    mocker.patch("shutil.copyfile", return_value=None)

    # Mock subprocess.Popen to simulate an external call to `mlp` command
    mock_popen = mocker.patch("subprocess.Popen")
    mock_popen.return_value.__enter__.return_value.communicate.return_value = (b'', b'')  # stdout, stderr
    mock_popen.return_value.__enter__.return_value.returncode = mock_subprocess

    # Initialize MTPWithMLIP3 with mock parameters
    model = MTPWithMLIP3(mlip_path="/mock/path", name="test_model")
    # Call the train method


    return_code = model.train(
        train_structures=[MockStructure(['H', 'O']), MockStructure(['Si'])],
        train_energies=[1, 2],
        train_forces=[],
        train_stresses=[],
        unfitted_mtp="08.almtp",
        fitted_mtp_savedir="/mock/dir"
    )
    # Assert the expected results
    assert return_code == mock_subprocess, "The train method should return the mocked subprocess return code."

    # Optionally, assert that mocked methods were called the expected number of times
    model.write_cfg.assert_called()
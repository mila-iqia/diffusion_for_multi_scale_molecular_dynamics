import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pymatgen.core import Structure
from sklearn.metrics import mean_absolute_error

from diffusion_for_multi_scale_molecular_dynamics.models.mlip.mtp import (
    MTPArguments, MTPWithMLIP3)
from diffusion_for_multi_scale_molecular_dynamics.models.mlip.utils import (
    MLIPInputs, extract_energy_from_thermo_log,
    extract_structure_and_forces_from_file, get_metrics_from_pred,
    prepare_mlip_inputs_from_lammps)


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
    mock_popen.return_value.__enter__.return_value.communicate.return_value = (
        b"",
        b"",
    )  # stdout, stderr
    mock_popen.return_value.__enter__.return_value.returncode = 0
    return mock_popen


# Mock the external dependencies and method calls within the MTPWithMLIP3.train method
def test_train(mocker, mock_popen, tmpdir):
    # Mock os.path.exists to always return True
    mocker.patch("os.path.exists", return_value=True)

    # Mock check_structures_forces_stresses to return a value without needing real input
    mocker.patch(
        "diffusion_for_multi_scale_molecular_dynamics.models.mlip.mtp.check_structures_forces_stresses",
        side_effect=passthrough,
    )

    # Mock pool_from to return a simplified pool object
    mocker.patch(
        "diffusion_for_multi_scale_molecular_dynamics.models.mlip.mtp.pool_from",
        return_value="simple_pool_object",
    )

    # Mock self.write_cfg to simulate creating a config file without file operations
    mocker.patch.object(MTPWithMLIP3, "write_cfg", return_value="mock_filename.cfg")

    mocker.patch("shutil.copyfile", return_value=None)

    # Initialize MTPWithMLIP3 with mock parameters
    mtp_args = MTPArguments(
        mlip_path="/mock/path",
        name="test_model",
        unfitted_mtp="08.almtp",
        fitted_mtp_savedir=tmpdir,
    )
    model = MTPWithMLIP3(mtp_args)
    # Call the train method
    mtp_inputs = MLIPInputs(
        structure=[FakeStructure(["H", "O"]), FakeStructure(["Si"])],
        forces=[],
        energy=[1, 2],
    )

    _ = model.train(
        mtp_inputs,
    )

    # Assert that mocked methods were called
    model.write_cfg.assert_called()


@pytest.fixture
def fake_structure():
    # Setup a simple mock structure object
    # Replace with appropriate structure setup for your use case
    return Structure(
        lattice=[1, 0, 0, 0, 1, 0, 0, 0, 1], species=[""], coords=[[0, 0, 0]]
    )


@pytest.fixture
def mtp_instance(mocker):
    # Mock __init__ to not execute its original behavior
    mocker.patch.object(MTPWithMLIP3, "__init__", lambda x, y: None)
    # Setup a mocked instance with necessary attributes
    instance = MTPWithMLIP3("mock_path")
    instance.mlp_command = "mock_mlp_command"
    instance.fitted_mtp = "mock_fitted_mtp"
    instance.elements = ["Si"]
    return instance


def test_evaluate(mocker, fake_structure, mtp_instance, mock_popen):
    test_structures = [fake_structure]
    test_energies = [1.0]
    test_forces = [[[0, 0, 0]]]

    # Mock check_structures_forces_stresses to return the arguments unmodified
    mocker.patch(
        "diffusion_for_multi_scale_molecular_dynamics.models.mlip.mtp.check_structures_forces_stresses",
        side_effect=lambda s, f, st: (s, f, st),
    )

    # Mock pool_from to return a mocked value
    mocker.patch(
        "diffusion_for_multi_scale_molecular_dynamics.models.mlip.mtp.pool_from",
        return_value="mock_pool",
    )

    # Mock self.write_cfg to simulate creating a config file without file operations
    mocker.patch.object(MTPWithMLIP3, "write_cfg", return_value="mock_filename.cfg")

    # Mock read_cfgs to simulate reading of configurations without accessing the file system
    mocker.patch.object(MTPWithMLIP3, "read_cfgs", return_value="mock_dataframe")

    # Mock os.remove, shutil.copyfile and os.path.exists since evaluate interacts with the filesystem
    mocker.patch("os.remove", return_value=None)
    mocker.patch("shutil.copyfile", return_value=None)
    mocker.patch("os.path.exists", return_value=True)

    mtp_inputs = MLIPInputs(
        structure=test_structures, forces=test_forces, energy=test_energies
    )

    # Perform the test
    df_predict = mtp_instance.evaluate(mtp_inputs)

    # Assertions can vary based on the real output of `read_cfgs`
    # Here's an example assertion assuming `read_cfgs` returns a string in this mocked scenario
    assert df_predict == "mock_dataframe", (
        "Evaluate method should return mock" + "dataframes"
    )


def test_read_cfgs(mtp_instance):
    cfg_path = Path(__file__).parent.joinpath("mtp_cfg_examples.txt")
    df = mtp_instance.read_cfgs(cfg_path, True)
    assert np.array_equal(df["x"], [0.1, 0.2, 0.3])
    assert np.array_equal(df["y"], [1.1, 1.2, 1.3])
    assert np.array_equal(df["z"], [2.1, 2.2, 2.3])
    assert np.array_equal(df["fx"], [3.1, 3.2, 3.3])
    assert np.array_equal(df["fy"], [4.1, 4.2, 4.3])
    assert np.array_equal(df["fz"], [5.1, 5.2, 5.3])
    assert np.array_equal(df["nbh_grades"], [6.1, 6.2, 6.3])


def test_extract_structure_and_forces_from_file(tmpdir):
    # test the function reading the lammps files.
    # TODO refactor to move to data/
    # Create a mock LAMMPS output
    yaml_content = {
        "box": [[0, 10], [0, 10], [0, 10]],  # x_lim, y_lim, z_lim
        "keywords": ["x", "y", "z", "element", "fx", "fy", "fz"],
        "data": [[1, 1, 1, 1, 0.1, 0.2, 0.3], [2, 2, 2, 2, 0.4, 0.5, 0.6]],
    }
    yaml_file = os.path.join(tmpdir, "lammps.yaml")
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    # Mock atom dict that the function expects
    atom_dict = {1: "H", 2: "He"}

    # Call the function
    structures, forces = extract_structure_and_forces_from_file(yaml_file, atom_dict)

    # Verify structures
    assert isinstance(structures, list)
    assert len(structures) == 1
    assert all(isinstance(structure, Structure) for structure in structures)

    # Verify the lattice was set up correctly, assuming a simple cubic lattice
    assert np.allclose(structures[0].lattice.matrix, np.diag([10, 10, 10]))

    # Verify species and positions
    species = structures[0].species
    assert [str(s) for s in species] == ["H", "He"]
    # frac coordinates are reduced coordinates - the values in data are cartesian coordinates
    # divide by the box length (10) to convert
    np.testing.assert_array_almost_equal(
        structures[0].frac_coords, [[1 / 10, 1 / 10, 1 / 10], [2 / 10, 2 / 10, 2 / 10]]
    )

    # Verify forces
    assert isinstance(forces, list)
    assert len(forces) == 1
    assert len(forces[0]) == 2  # Two sets of forces for two atoms
    expected_forces = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert np.allclose(forces[0], expected_forces)


def test_extract_energy_from_thermo_log(tmpdir):
    # test the function reading the lammps thermodynamic output files.
    # TODO refactor to move to data/
    # Create a mock LAMMPS thermodynamic output
    log_content = """
    keywords:
      - Step
      - KinEng
      - PotEng
    data:
      - [0, 50.5, -100.0]
      - [1, 51.0, -101.5]
    """
    yaml_path = os.path.join(tmpdir, "thermo.yaml")
    with open(yaml_path, "w") as f:
        f.write(log_content)

    # Call the function
    energies = extract_energy_from_thermo_log(yaml_path)

    # Check the results
    expected_energies = [-49.5, -50.5]  # KinEng + PotEng for each step
    assert isinstance(energies, list)
    assert energies == expected_energies


@pytest.fixture
def mock_extract_energy_from_thermo_log(mocker):
    return mocker.patch(
        "diffusion_for_multi_scale_molecular_dynamics.models.mlip.utils.extract_energy_from_thermo_log",
        return_value=[],
    )


@pytest.fixture
def mock_extract_structure_and_forces(mocker):
    return mocker.patch(
        "diffusion_for_multi_scale_molecular_dynamics.models.mlip.utils.extract_structure_and_forces_from_file",
        return_value=([], []),
    )


def test_prepare_mtp_inputs_from_lammps(
    mock_extract_structure_and_forces, mock_extract_energy_from_thermo_log, tmpdir
):
    # Create mock file paths
    output_yaml_files = [
        os.path.join(tmpdir, "output1.yaml"),
        os.path.join(tmpdir, "output2.yaml"),
    ]
    thermo_yaml_files = [
        os.path.join(tmpdir, "thermo1.yaml"),
        os.path.join(tmpdir, "thermo2.yaml"),
    ]

    # Mock atom dictionary
    atom_dict = {1: "H", 2: "He"}

    # Call the function
    mtp_inputs = prepare_mlip_inputs_from_lammps(
        output_yaml_files, thermo_yaml_files, atom_dict
    )

    # Verify that the mocks were called correctly
    assert mock_extract_structure_and_forces.call_count == 2
    mock_extract_structure_and_forces.assert_called_with(
        output_yaml_files[1], atom_dict, True
    )

    assert mock_extract_energy_from_thermo_log.call_count == 2
    mock_extract_energy_from_thermo_log.assert_called_with(thermo_yaml_files[1])

    # Verify that the result is correctly structured
    assert isinstance(mtp_inputs.structure, list)
    assert isinstance(mtp_inputs.energy, list)
    assert isinstance(mtp_inputs.forces, list)

    # Verify that the data from the mocks is aggregated into the results correctly
    assert mtp_inputs.structure == mock_extract_structure_and_forces.return_value[
        0
    ] * len(output_yaml_files)
    assert mtp_inputs.forces == mock_extract_structure_and_forces.return_value[1] * len(
        output_yaml_files
    )
    assert mtp_inputs.energy == mock_extract_energy_from_thermo_log.return_value * len(
        thermo_yaml_files
    )


def test_get_metrics_from_pred():
    # test function from train_mtp
    # TODO get better metrics and refactor the script
    # Assuming there are 2 structures, each with 2 atoms (Total 4 readings)
    df_orig = pd.DataFrame(
        {
            "structure_index": [0, 0, 1, 1],
            "atom_index": [0, 1, 0, 1],
            "energy": [
                1,
                1,
                3,
                3,
            ],  # Total energy for the structure is the same for both atoms
            "fx": [0.1, -0.1, 0.2, -0.2],
            "fy": [0.1, -0.1, 0.2, -0.2],
            "fz": [0.1, -0.1, 0.2, -0.2],
        }
    )

    # Predicted data with some error introduced
    df_predict = pd.DataFrame(
        {
            "structure_index": [0, 0, 1, 1],
            "atom_index": [0, 1, 0, 1],
            "energy": [
                1.1,
                1.1,
                3.3,
                3.3,
            ],  # energy cannot be different per atom for a given structure
            "fx": [0.12, -0.08, 0.23, -0.17],
            "fy": [0.09, -0.11, 0.19, -0.21],
            "fz": [0.11, -0.09, 0.18, -0.22],
        }
    )

    # Calculate expected MAE for energy and forces
    # 1 value per structure - here, we take indices 0 and 2
    expected_mae_energy = (
        np.abs(
            df_orig["energy"].iloc[[0, 2]] - df_predict["energy"].iloc[[0, 2]]
        ).mean()
        / 2
    )
    # we take the energy per atom. 2 atoms per structure here, so we can simply divide by 2

    expected_mae_forces = mean_absolute_error(
        df_orig[["fx", "fy", "fz"]].values.flatten(),
        df_predict[["fx", "fy", "fz"]].values.flatten(),
    )

    # Call the function under test
    mae_energy, mae_forces = get_metrics_from_pred(df_orig, df_predict)

    # Verify the MAE values are as expected
    assert mae_energy == expected_mae_energy
    assert mae_forces == expected_mae_forces

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms

from diffusion_for_multi_scale_molecular_dynamics.models.mlip.ace import (
    ACE_MLIP, ACE_arguments)


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


@pytest.fixture
def num_samples():
    return 8


@pytest.fixture
def energies(num_samples):
    return np.random.random(num_samples)


@pytest.fixture
def num_atoms():
    return 5


@pytest.fixture
def ase_atoms(num_atoms, num_samples):
    positions = np.random.random((num_atoms, 3))
    cell = np.random.uniform(5, 10, (3,))
    return [
        Atoms(f"Si{num_atoms}", positions=positions, cell=cell, pbc=[1, 1, 1])
        for _ in range(num_samples)
    ]


@pytest.fixture
def forces(num_samples, num_atoms):
    return np.random.random((num_samples, num_atoms, 3))


@pytest.fixture
def ace_dataset(num_samples, energies, forces, num_atoms, ase_atoms):
    df = {
        "energy": energies,
        "energy_corrected": energies,
        "forces": [f for f in forces],
        "NUMBER_OF_ATOMS": [num_atoms for _ in range(num_samples)],
        "ase_atoms": ase_atoms,
    }
    return pd.DataFrame(df)


@pytest.mark.not_on_github
# this test is slow and requires pacemaker in the CLI - not included with pip
def test_train(ace_dataset, tmpdir):
    # Initialize ACE_MLIP with mock parameters
    ace_args = ACE_arguments(
        config_path=Path(__file__).parent / "ace_config.yaml",
        fitted_ace_savedir=os.path.join(tmpdir, "fitted_ace"),
        working_dir=os.path.join(tmpdir, "working_dir"),
    )
    model = ACE_MLIP(ace_args)
    # Call the train method
    returned_potential = model.train(
        ace_dataset,
    )

    assert os.path.exists(returned_potential)


@pytest.mark.not_on_github
def test_evaluate(ace_dataset, num_atoms, tmpdir):
    ace_args = ACE_arguments(
        config_path=Path(__file__).parent / "ace_config.yaml",
        fitted_ace_savedir=os.path.join(tmpdir, "fitted_ace"),
        working_dir=os.path.join(tmpdir, "working_dir"),
    )
    model = ACE_MLIP(ace_args)
    # Call the train method

    returned_eval_data = model.evaluate(
        ace_dataset, mlip_name="ace_fitted.yaml", mode="train_and_eval"
    )

    positions = np.stack([x.get_positions() for x in ace_dataset["ase_atoms"]])

    for i, var_name in enumerate(["x", "y", "z"]):
        returned_pos = returned_eval_data[var_name]
        assert np.allclose(returned_pos, positions[:, :, i].flatten())

    forces = np.stack(ace_dataset["forces"]).flatten()
    returned_forces = (
        returned_eval_data[["fx_target", "fy_target", "fz_target"]].to_numpy().flatten()
    )
    assert np.allclose(returned_forces, forces)

    energy = np.concatenate([[e] * num_atoms for e in ace_dataset["energy_corrected"]])
    assert np.allclose(energy, returned_eval_data["energy_target"])

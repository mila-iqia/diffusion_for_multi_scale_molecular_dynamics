"""Unit tests for outputs to ovito."""

import os

import numpy as np
import pandas as pd
import pytest

from diffusion_for_multi_scale_molecular_dynamics.analysis.ovito_utilities.ovito_visualisation import (
    get_lattice_from_lammps, mtp_predictions_to_ovito)


class TestMTP2Ovito:
    """This class tests the functions to convert outputs to OVITO readable files."""

    @pytest.fixture
    def lammps_output(self, tmpdir):
        lattice_output = """
box:
    - [0, 4.0]
    - [0, 5.0]
    - [0, 6.0]
---
box:
    - [0, 4.0]
    - [0, 5.0]
    - [0, 6.0]
        """
        with open(os.path.join(tmpdir, "test_output.yaml"), "w") as f:
            f.write(lattice_output)
        return os.path.join(tmpdir, "test_output.yaml")

    def test_get_lattice_from_lammps(self, lammps_output):
        # Get the lattice array from the function using the test file
        lattice = get_lattice_from_lammps(lammps_output)

        expected_lattice = np.array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])

        # Assert that the returned lattice is as expected
        np.testing.assert_array_equal(lattice, expected_lattice)

    @pytest.fixture
    def fake_prediction_csv(self, tmpdir):
        # Create a DataFrame with mock prediction data
        df = pd.DataFrame(
            {
                "structure_index": [1, 1, 2, 2],
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [0.1, 1.1, 2.1, 3.1],
                "z": [0.2, 1.2, 2.2, 3.2],
                "nbh_grades": [0.3, 1.3, 2.3, 3.3],
            }
        )
        file_path = os.path.join(tmpdir, "mock_predictions.csv")
        df.to_csv(file_path, index=False)

        return file_path

    def test_mtp_predictions_to_ovito(self, fake_prediction_csv, tmpdir):
        # Define the lattice
        lattice = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        # output file name
        output_name = os.path.join(tmpdir, "test_output")
        # Run the conversion function
        mtp_predictions_to_ovito(fake_prediction_csv, lattice, output_name)

        # check the output exists
        assert os.path.exists(os.path.join(tmpdir, "test_output.xyz"))

        with open(output_name + ".xyz", "r") as f:
            lines = f.readlines()

        # Verify that the number of lines corresponds to
        # 1 line for number of atoms + 1 line for lattice + 1 line per atom = 6 lines per structure
        assert len(lines) == (1 + 1 + 2) * 2  # 2 structures

        assert int(lines[0]) == int(lines[len(lines) // 2]) == 2

        expected_lattice_line = (
            'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Origin="0 0 0" '
        )
        expected_lattice_line += 'pbc="T T T" Properties=pos:R:3:MaxVolGamma:R:1\n'
        assert lines[1] == lines[len(lines) // 2 + 1] == expected_lattice_line

        # check the atom values
        for x, y in enumerate([2, 3, 6, 7]):
            assert np.array_equal(
                list(map(float, lines[y].split())), [x, x + 0.1, x + 0.2, x + 0.3]
            )

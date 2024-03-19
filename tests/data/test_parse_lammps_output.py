import os

import numpy as np
import pandas as pd
import pytest
import yaml

from crystal_diffusion.data.parse_lammps_outputs import parse_lammps_output


def generate_fake_yaml(filename, documents, multiple_docs=True):
    # Write the YAML content
    with open(filename, 'w') as yaml_file:
        if multiple_docs:
            yaml.dump_all(documents, yaml_file)
        else:
            yaml.dump(documents, yaml_file)


@pytest.fixture
def fake_lammps_yaml(tmpdir):
    # fake LAMMPS output file with 4 MD steps in 1D for 3 atoms
    yaml_content = [
        {'keywords': ['id', 'type', 'x', 'fx'],
         'data': [[0, 1, 0.1, 0.01], [1, 2, 0.2, 0.02], [2, 1, 0.3, 0.03]],
         'box': [[0, 0.6], [0, 1.6], [0, 2.6]]},
        {'keywords': ['id', 'type', 'x', 'fx'],
         'data': [[0, 1, 1.1, 1.01], [1, 2, 1.2, 1.02], [2, 1, 1.3, 1.03]],
         'box': [[0, 0.6], [0, 1.6], [0, 2.6]]},
        {'keywords': ['id', 'type', 'x', 'fx'],
         'data': [[0, 1, 2.1, 2.01], [1, 2, 2.2, 2.02], [2, 1, 2.3, 2.03]],
         'box': [[0, 0.6], [0, 1.6], [0, 2.6]]},
        {'keywords': ['id', 'type', 'x', 'fx'],
         'data': [[0, 1, 3.1, 3.01], [1, 2, 3.2, 3.02], [2, 1, 3.3, 3.03]],
         'box': [[0, 0.6], [0, 1.6], [0, 2.6]]},
    ]
    file = os.path.join(tmpdir, 'fake_lammps_dump.yaml')
    generate_fake_yaml(file, yaml_content)
    return file


@pytest.fixture
def fake_thermo_yaml(tmpdir):
    # fake LAMMPS thermo file with 4 MD steps
    yaml_content = {
        'keywords': ['KinEng', 'PotEng'],
        'data': [[0.4, 0.5], [1.4, 1.5], [2.4, 2.5], [3.4, 3.5]],
    }
    file = os.path.join(tmpdir, 'fake_lammps_thermo.yaml')
    generate_fake_yaml(file, yaml_content, multiple_docs=False)
    return file


def test_parse_lammps_outputs(fake_lammps_yaml, fake_thermo_yaml, tmpdir):
    output_name = os.path.join(tmpdir, 'test.parquet')
    parse_lammps_output(fake_lammps_yaml, fake_thermo_yaml, output_name)
    # check that a file exists
    assert os.path.exists(output_name)

    df = pd.read_parquet(output_name)
    assert not df.empty

    assert len(df) == 4

    for i, v in enumerate(['id', 'type', 'x', 'fx', 'energy', 'box']):
        assert v in df.keys()
        for x in range(4):
            if v == 'id':
                assert np.array_equal(df[v][x], [0, 1, 2])
            elif v == 'type':
                assert np.array_equal(df[v][x], [1, 2, 1])
            elif v == 'x':
                assert np.allclose(df[v][x], [x + 0.1 * y for y in range(1, 4)])
            elif v == 'fx':
                assert np.allclose(df[v][x], [x + 0.01 * y for y in range(1, 4)])
            elif v == 'energy':
                assert np.allclose(df[v][x], [2 * x + 0.9])
            else:  # v == 'box'
                assert np.allclose(df[v][x], [x + 0.6 for x in range(3)])

"""Test the train_diffusion.py script.

The goal of this test module is to "smoke test" the train_diffusion.py script. That is to say, we want
to make sure that the code can run without crashing when given appropriate inputs: we are not testing
that the results are correct.
"""
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from crystal_diffusion import train_diffusion


def get_fake_configuration_dataframe(number_of_configurations, number_of_atoms):
    """Get fake configuration dataframe.

    Generate a random, fake configuration dataframe with the minimal columns to reproduce the
    behavior of 'parse_lammps_output'.
    """
    spatial_dimension = 3
    list_rows = []
    for row_id in range(number_of_configurations):
        box = 10 * np.random.rand(spatial_dimension)

        row = dict(box=box,
                   x=box[0] * np.random.rand(number_of_atoms),
                   y=box[1] * np.random.rand(number_of_atoms),
                   z=box[2] * np.random.rand(number_of_atoms),
                   type=np.random.randint(0, 10, number_of_atoms)
                   )

        list_rows.append(row)

    df = pd.DataFrame(list_rows)
    return df


@pytest.fixture()
def number_of_atoms():
    return 8


@pytest.fixture()
def config(number_of_atoms):
    data_config = dict(batch_size=4, num_workers=0, max_atom=number_of_atoms)

    model_config = dict(score_network={'hidden_dimensions': [16, 16]},
                        noise={'total_time_steps': 10})

    optimizer_config = dict(name='adam', learning_rate=0.001)

    early_stopping_config = dict(metric='val_loss', mode='min', patience=3)

    config = dict(loss='cross_entropy',
                  max_epoch=5,
                  exp_name='smoke_test',
                  seed=1234,
                  spatial_dimension=3,
                  data=data_config,
                  model=model_config,
                  optimizer=optimizer_config,
                  early_stopping=early_stopping_config)

    return config


@pytest.fixture()
def paths(tmpdir, config, number_of_atoms):
    np.random.seed(1234)

    paths = dict()
    for directory_name in ['data', 'processed_datadir', 'dataset_working_dir', 'tmp-folder', 'output']:
        # use random names to make sure we didn't accidentally hardcode a folder name.
        directory = os.path.join(tmpdir, f"folder{np.random.randint(99999999)}")
        paths[directory_name] = directory
        os.makedirs(directory)

    # dump some fake data into the appropriate folder.
    raw_data_directory = paths['data']
    for mode in ['train', 'valid']:
        for run in [1, 2, 3]:
            df = get_fake_configuration_dataframe(number_of_configurations=10,
                                                  number_of_atoms=number_of_atoms)

            run_directory = os.path.join(raw_data_directory, f'{mode}_run_{run}')
            os.makedirs(run_directory)
            df.to_pickle(os.path.join(run_directory, 'fake_data.pickle'))

    paths['log'] = os.path.join(tmpdir, f"file{np.random.randint(99999999)}.log")
    paths['config'] = os.path.join(tmpdir, f"file{np.random.randint(99999999)}.yaml")

    with open(paths['config'], 'w') as fd:
        yaml.dump(config, fd)

    return paths


@pytest.fixture
def mock_get_lammps_output_method(monkeypatch):
    # Reading in yaml files is slow. This mock aims to go around this.

    def path_to_fake_data_pickle(self, run_dir):
        fake_data_path = os.path.join(run_dir, 'fake_data.pickle')
        return pd.read_pickle(fake_data_path)

    monkeypatch.setattr(
        'crystal_diffusion.data.diffusion.data_preprocess.LammpsProcessorForDiffusion.get_lammps_output',
        path_to_fake_data_pickle
    )


@pytest.fixture()
def args(paths, config):
    """Input arguments for main."""
    input_args = [f"--log={paths['log']}",
                  f"--config={paths['config']}",
                  f"--data={paths['data']}",
                  f"--processed_datadir={paths['processed_datadir']}",
                  f"--dataset_working_dir={paths['dataset_working_dir']}",
                  f"--tmp-folder={paths['tmp-folder']}",
                  f"--output={paths['output']}",
                  f"--accelerator={'cpu'}",
                  f"--devices={1}"]

    return input_args


def test_main(args, mock_get_lammps_output_method):
    train_diffusion.main(args)

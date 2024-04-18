"""Test the train_diffusion.py script.

The goal of this test module is to "smoke test" the train_diffusion.py script. That is to say, we want
to make sure that the code can run without crashing when given appropriate inputs: we are not testing
that the results are correct.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import pytest
import yaml

from crystal_diffusion import train_diffusion
from crystal_diffusion.callbacks.callback_loader import create_all_callbacks
from crystal_diffusion.callbacks.standard_callbacks import (BEST_MODEL_NAME,
                                                            LAST_MODEL_NAME)

best_model_regex = re.compile(r"best_model-epoch=(?P<epoch>(\d+)).*.ckpt")
last_model_regex = re.compile(r"last_model-epoch=(?P<epoch>(\d+)).*.ckpt")


class DelayedInterrupt:
    """This class will raise a KeyboardInterrupt when its method is called a certain number of times."""
    def __init__(self, how_many_epochs_before_interrupt: int):
        self.how_many_epochs_before_interrupt = how_many_epochs_before_interrupt
        self.counter = 0

    def delayed_interrupt(self):
        self.counter += 1
        if self.counter == self.how_many_epochs_before_interrupt + 1:
            # Only interrupt once.
            raise KeyboardInterrupt


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
                   type=np.random.randint(0, 10, number_of_atoms),
                   energy=np.random.rand()
                   )

        list_rows.append(row)

    df = pd.DataFrame(list_rows)
    return df


def get_config(number_of_atoms: int, max_epoch: int):
    data_config = dict(batch_size=4, num_workers=0, max_atom=number_of_atoms)

    model_config = dict(score_network={'n_hidden_dimensions': 2,
                                       'hidden_dimensions_size': 16},
                        noise={'total_time_steps': 10})

    optimizer_config = dict(name='adam', learning_rate=0.001)

    sampling_dict = {'spatial_dimension': 3,
                     'number_of_corrector_steps': 1,
                     'number_of_atoms': number_of_atoms,
                     'number_of_samples': 4,
                     'sample_every_n_epochs': 1,
                     'cell_dimensions': [1.23, 4.56, 7.89]}

    early_stopping_config = dict(metric='validation_epoch_loss', mode='min', patience=max_epoch)
    model_checkpoint_config = dict(monitor='validation_epoch_loss', mode='min')
    diffusion_sampling_config = dict(noise={'total_time_steps': 10}, sampling=sampling_dict)

    config = dict(max_epoch=max_epoch,
                  exp_name='smoke_test',
                  seed=9999,
                  spatial_dimension=3,
                  data=data_config,
                  model=model_config,
                  optimizer=optimizer_config,
                  early_stopping=early_stopping_config,
                  model_checkpoint=model_checkpoint_config,
                  diffusion_sampling=diffusion_sampling_config,
                  logging=['csv', 'tensorboard'])
    return config


@pytest.fixture(scope="module")
def random_number_generator():
    # Create a random number generator to decouple randomness in the test
    # from the randomness in the model. The model uses its own seeds, which will
    # affect the behavior of numpy.random: leveraging an independent rng will insure
    # control over test randomness.
    rng = np.random.default_rng(seed=2345234234)
    return rng


@pytest.fixture()
def number_of_atoms():
    return 8


@pytest.fixture()
def max_epoch():
    return 5


@pytest.fixture()
def config(number_of_atoms, max_epoch):
    return get_config(number_of_atoms, max_epoch=max_epoch)


@pytest.fixture()
def paths(tmpdir, config, number_of_atoms, random_number_generator):
    paths = dict()
    for directory_name in ['data', 'processed_datadir', 'dataset_working_dir', 'tmp-folder', 'output']:
        # use random names to make sure we didn't accidentally hardcode a folder name.
        directory = os.path.join(tmpdir, f"folder{random_number_generator.integers(99999999)}")
        paths[directory_name] = directory

    # dump some fake data into the appropriate folder.
    raw_data_directory = paths['data']
    for mode in ['train', 'valid']:
        for run in [1, 2, 3]:
            df = get_fake_configuration_dataframe(number_of_configurations=10,
                                                  number_of_atoms=number_of_atoms)

            run_directory = os.path.join(raw_data_directory, f'{mode}_run_{run}')
            os.makedirs(run_directory)
            df.to_pickle(os.path.join(run_directory, 'fake_data.pickle'))

    paths['config'] = os.path.join(tmpdir, f"file{random_number_generator.integers(99999999)}.yaml")

    with open(paths['config'], 'w') as fd:
        yaml.dump(config, fd)

    return paths


@pytest.fixture()
def args(paths):
    """Input arguments for main."""
    input_args = [f"--config={paths['config']}",
                  f"--data={paths['data']}",
                  f"--processed_datadir={paths['processed_datadir']}",
                  f"--dataset_working_dir={paths['dataset_working_dir']}",
                  f"--tmp-folder={paths['tmp-folder']}",
                  f"--output={paths['output']}",
                  f"--accelerator={'cpu'}",
                  f"--devices={1}"]

    return input_args


@pytest.fixture()
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
def callback_dictionary(config, paths):
    return create_all_callbacks(hyper_params=config, output_directory=paths['output'], verbose=False)


def test_checkpoint_callback(args, paths, mock_get_lammps_output_method, max_epoch):
    train_diffusion.main(args)
    best_model_path = os.path.join(paths['output'], BEST_MODEL_NAME)
    last_model_path = os.path.join(paths['output'], LAST_MODEL_NAME)

    model_paths = [best_model_path, last_model_path]
    regexes = [best_model_regex, last_model_regex]
    should_test_epoch_number = [False, True]  # the 'best' model epoch is ill-defined. Don't test!

    for model_path, regex, test_epoch_number in zip(model_paths, regexes, should_test_epoch_number):
        paths_in_directory = glob.glob(model_path + '/*.ckpt')

        assert len(paths_in_directory) == 1
        model_filename = os.path.basename(paths_in_directory[0])
        match_object = regex.match(model_filename)
        assert match_object is not None
        if test_epoch_number:
            model_epoch = int(match_object.group('epoch'))
            assert model_epoch == max_epoch - 1  # the epoch counter starts at zero!


def test_restart(args, paths, mock_get_lammps_output_method, max_epoch, mocker):
    last_model_path = os.path.join(paths['output'], LAST_MODEL_NAME)

    method_to_patch = ("crystal_diffusion.models.position_diffusion_lightning_model."
                       "PositionDiffusionLightningModel.on_train_epoch_start")

    interruption_epoch = max_epoch // 2
    interruptor = DelayedInterrupt(how_many_epochs_before_interrupt=interruption_epoch)
    mocker.patch(method_to_patch, side_effect=interruptor.delayed_interrupt)

    train_diffusion.main(args)

    # Check that there is only one last model, and that it fits expectation
    paths_in_directory = glob.glob(last_model_path + '/*.ckpt')
    last_model_filename = os.path.basename(paths_in_directory[0])
    match_object = last_model_regex.match(last_model_filename)
    assert match_object is not None
    model_epoch = int(match_object.group('epoch'))
    assert model_epoch == interruption_epoch - 1

    # Restart training
    train_diffusion.main(args)

    # test that training was restarted.
    # NOTE: the checkpoints produced before the interruption are NOT
    # automatically deleted by pytorch-lightning. This means that there
    # may be multiple "last" and "best" checkpoints in their respective folders,
    # which can be mildly confusing. It would be dangerous at this time to
    # implement complex (and fragile) logic to delete stale checkpoints, and risk
    # deleting all checkpoints. Interruptions are rare: it is best to deal with
    # multiple checkpoints by trusting the largest epoch number.
    list_epoch_numbers = []
    for path in glob.glob(last_model_path + '/*.ckpt'):
        last_model_filename = os.path.basename(path)
        match_object = last_model_regex.match(last_model_filename)
        assert match_object is not None
        model_epoch = int(match_object.group('epoch'))
        list_epoch_numbers.append(model_epoch)
    assert np.max(list_epoch_numbers) == max_epoch - 1

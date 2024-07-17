"""Test the train_diffusion.py script.

The goal of this test module is to "smoke test" the train_diffusion.py script. That is to say, we want
to make sure that the code can run without crashing when given appropriate inputs: we are not testing
that the results are correct.
"""
import glob
import os
import re
from typing import Union

import numpy as np
import pytest
import yaml

from crystal_diffusion import train_diffusion
from crystal_diffusion.callbacks.standard_callbacks import (BEST_MODEL_NAME,
                                                            LAST_MODEL_NAME)
from tests.conftest import TestDiffusionDataBase

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


def get_prediction_head_parameters(name: str):
    if name == 'mlp':
        head_parameters = dict(name='mlp',
                               hidden_dimensions_size=16,
                               n_hidden_dimensions=3)

    elif name == 'equivariant':
        head_parameters = dict(name='equivariant',
                               time_embedding_irreps="16x0e",
                               number_of_layers=2,
                               gate="silu")
    else:
        raise NotImplementedError("This score network is not implemented")

    return head_parameters


def get_score_network(architecture: str, head_name: Union[str, None], number_of_atoms: int):
    if architecture == 'mlp':
        assert head_name is None, "There are no head options for a MLP score network."
        score_network = dict(architecture='mlp',
                             number_of_atoms=number_of_atoms,
                             n_hidden_dimensions=2,
                             hidden_dimensions_size=16)
    elif architecture == 'mace':
        score_network = dict(architecture='mace',
                             r_max=3.0,
                             num_bessel=4,
                             hidden_irreps="8x0e + 8x1o",
                             number_of_atoms=number_of_atoms,
                             radial_MLP=[4, 4, 4],
                             prediction_head_parameters=get_prediction_head_parameters(head_name))

    elif architecture == 'diffusion_mace':
        assert head_name is None, "There are no head options for a MLP score network."
        score_network = dict(architecture='diffusion_mace',
                             r_max=3.0,
                             num_bessel=4,
                             hidden_irreps="8x0e + 8x1o",
                             mlp_irreps="8x0e",
                             number_of_mlp_layers=1,
                             number_of_atoms=number_of_atoms,
                             radial_MLP=[4, 4, 4])

    elif architecture == 'egnn':
        score_network = dict(architecture='egnn',
                             hidden_dimensions_size=32,
                             number_of_layers=3)
    else:
        raise NotImplementedError("This score network is not implemented")
    return score_network


def get_config(number_of_atoms: int, max_epoch: int, architecture: str, head_name: Union[str, None],
               sampling_algorithm: str):
    data_config = dict(batch_size=4, num_workers=0, max_atom=number_of_atoms)

    model_config = dict(score_network=get_score_network(architecture, head_name, number_of_atoms),
                        loss={'algorithm': 'mse'},
                        noise={'total_time_steps': 10})

    optimizer_config = dict(name='adam', learning_rate=0.001)
    scheduler_config = dict(name='ReduceLROnPlateau', factor=0.6, patience=2)

    sampling_dict = dict(algorithm=sampling_algorithm,
                         spatial_dimension=3,
                         number_of_atoms=number_of_atoms,
                         number_of_samples=4,
                         sample_every_n_epochs=1,
                         record_samples=True,
                         cell_dimensions=[10., 10., 10.])
    if sampling_algorithm == 'predictor_corrector':
        sampling_dict["number_of_corrector_steps"] = 1

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
                  scheduler=scheduler_config,
                  early_stopping=early_stopping_config,
                  model_checkpoint=model_checkpoint_config,
                  loss_monitoring=dict(number_of_bins=10, sample_every_n_epochs=3),
                  diffusion_sampling=diffusion_sampling_config,
                  logging=['csv', 'tensorboard'])
    return config


@pytest.mark.parametrize("sampling_algorithm", ["ode", "predictor_corrector"])
@pytest.mark.parametrize("architecture, head_name",
                         [('egnn', None),
                          ('diffusion_mace', None),
                          ('mlp', None),
                          ('mace', 'equivariant'),
                          ('mace', 'mlp')])
class TestTrainDiffusion(TestDiffusionDataBase):
    @pytest.fixture()
    def max_epoch(self):
        return 5

    @pytest.fixture()
    def config(self, number_of_atoms, max_epoch, architecture, head_name, sampling_algorithm):
        return get_config(number_of_atoms, max_epoch=max_epoch,
                          architecture=architecture, head_name=head_name, sampling_algorithm=sampling_algorithm)

    @pytest.fixture()
    def all_paths(self, paths, tmpdir, config):
        all_paths = dict(data=paths['raw_data_dir'], processed_datadir=paths['processed_data_dir'])

        for directory_name in ['dataset_working_dir', 'output']:
            # use random names to make sure we didn't accidentally hardcode a folder name.
            directory = os.path.join(tmpdir, f"{directory_name}_{np.random.randint(99999999)}")
            all_paths[directory_name] = directory

        all_paths['config'] = os.path.join(tmpdir, f"config_{np.random.randint(99999999)}.yaml")

        with open(all_paths['config'], 'w') as fd:
            yaml.dump(config, fd)

        return all_paths

    @pytest.fixture()
    def args(self, all_paths, accelerator):
        """Input arguments for main."""
        input_args = [f"--config={all_paths['config']}",
                      f"--data={all_paths['data']}",
                      f"--processed_datadir={all_paths['processed_datadir']}",
                      f"--dataset_working_dir={all_paths['dataset_working_dir']}",
                      f"--output={all_paths['output']}",
                      f"--accelerator={accelerator}",
                      f"--devices={1}"]

        return input_args

    @pytest.mark.slow
    def test_checkpoint_callback(self, args, all_paths, max_epoch):
        train_diffusion.main(args)
        best_model_path = os.path.join(all_paths['output'], BEST_MODEL_NAME)
        last_model_path = os.path.join(all_paths['output'], LAST_MODEL_NAME)

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

    @pytest.mark.slow
    def test_restart(self, args, all_paths, max_epoch, mocker):
        last_model_path = os.path.join(all_paths['output'], LAST_MODEL_NAME)

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

        # Test that training was restarted.
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

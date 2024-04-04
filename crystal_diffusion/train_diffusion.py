"""Entry point to train a diffusion model."""
import argparse
import logging
import os
import shutil
import sys
import typing

import pytorch_lightning as pl
import yaml
from yaml import load

from crystal_diffusion.callbacks.callback_loader import create_all_callbacks
from crystal_diffusion.data.diffusion.data_loader import (
    LammpsForDiffusionDataModule, LammpsLoaderParameters)
from crystal_diffusion.main_utils import (MetricResult,
                                          get_crash_metric_result,
                                          get_optimized_metric_name_and_mode,
                                          report_to_orion_if_on)
from crystal_diffusion.models.model_loader import load_diffusion_model
from crystal_diffusion.utils.file_utils import rsync_folder
from crystal_diffusion.utils.hp_utils import check_and_log_hp
from crystal_diffusion.utils.logging_utils import log_exp_details
from crystal_diffusion.utils.reproducibility_utils import set_seed

logger = logging.getLogger(__name__)


def main(args: typing.Optional[typing.Any] = None):
    """Create and train a diffusion model: main entry point of the program.

    Note:
        This main.py file is meant to be called using the cli,
        see the `examples/local/run_diffusion.sh` file to see how to use it.
    """
    parser = argparse.ArgumentParser()
    # __TODO__ check you need all the following CLI parameters
    parser.add_argument('--log', help='log to this file (in addition to stdout/err)')
    parser.add_argument('--config',
                        help='config file with generic hyper-parameters,  such as optimizer, '
                             'batch_size, ... -  in yaml format')
    parser.add_argument('--data', help='path to a LAMMPS data set', required=True)
    parser.add_argument('--processed_datadir', help='path to the processed data directory', required=True)
    parser.add_argument('--dataset_working_dir', help='path to the Datasets working directory. Defaults to None',
                        default=None)
    parser.add_argument('--tmp-folder',
                        help='will use this folder as working folder - it will copy the input data '
                             'here, generate results here, and then copy them back to the output '
                             'folder')  # TODO possibly remove this
    parser.add_argument('--output', help='path to outputs - will store files here', required=True)
    parser.add_argument('--disable-progressbar', action='store_true',
                        help='will disable the progressbar while going over the mini-batch')
    parser.add_argument('--start-from-scratch', action='store_true',
                        help='will not load any existing saved model - even if present')
    parser.add_argument('--accelerator', help='PL trainer accelerator. Defaults to auto.', default='auto')
    parser.add_argument('--devices', default=1, help='pytorch-lightning devices kwarg. Defaults to 1.')
    parser.add_argument('--debug', action='store_true')  # TODO not used yet
    args = parser.parse_args(args)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if os.path.exists(args.output) and args.start_from_scratch:
        logger.info('Starting from scratch, removing any previous experiments.')
        shutil.rmtree(args.output)

    if os.path.exists(args.output):
        logger.info("Previous experiment found, resuming from checkpoint")
    else:
        os.makedirs(args.output)

    if args.tmp_folder is not None:
        # TODO data rsync to tmp_folder
        output_dir = os.path.join(args.tmp_folder, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = args.output

    # will log to a file if provided (useful for orion on cluster)
    if args.log is not None:
        handler = logging.handlers.WatchedFileHandler(args.log)
        formatter = logging.Formatter(logging.BASIC_FORMAT)
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)

    if args.config is not None:
        with open(args.config, 'r') as stream:
            hyper_params = load(stream, Loader=yaml.FullLoader)
    else:
        hyper_params = {}

    run(args, output_dir, hyper_params)

    if args.tmp_folder is not None:
        rsync_folder(output_dir + os.path.sep, args.output)


def run(args, output_dir, hyper_params):
    """Create and run the dataloaders, training loops, etc.

    Args:
        args (object): arguments passed from the cli
        output_dir (str): path to output folder
        hyper_params (dict): hyper parameters from the config file
    """
    # __TODO__ change the hparam that are used from the training algorithm
    # (and NOT the model - these will be specified in the model itself)
    logger.info('List of hyper-parameters:')
    check_and_log_hp(
        ['model', 'data', 'exp_name', 'max_epoch', 'optimizer', 'seed',
         'early_stopping'],
        hyper_params)

    if hyper_params["seed"] is not None:
        set_seed(hyper_params["seed"])

    log_exp_details(os.path.realpath(__file__), args)

    data_params = LammpsLoaderParameters(**hyper_params['data'])

    datamodule = LammpsForDiffusionDataModule(
        lammps_run_dir=args.data,
        processed_dataset_dir=args.processed_datadir,
        hyper_params=data_params,
        working_cache_dir=args.dataset_working_dir,
    )

    model = load_diffusion_model(hyper_params)

    try:
        metric_result = train(model=model,
                              datamodule=datamodule,
                              output=output_dir,
                              hyper_params=hyper_params,
                              use_progress_bar=not args.disable_progressbar,
                              accelerator=args.accelerator,
                              devices=args.devices)
        run_time_error = None
    except RuntimeError as err:
        run_time_error = err
        logger.error(err)
        metric_result = get_crash_metric_result(hyper_params)

    # clean up the data cache to save disk space
    datamodule.clean_up()

    report_to_orion_if_on(metric_result, run_time_error)


def train(model,
          datamodule,
          output: str,
          hyper_params: typing.Dict[typing.AnyStr, typing.Any],
          use_progress_bar: bool,
          accelerator=None,
          devices=None
          ):
    """Train a model: main training loop implementation.

    Args:
        model (obj): The neural network model object.
        datamodule (obj): lightning data module that will instantiate data loaders.
        output (str): Output directory.
        hyper_params (dict): Dict containing hyper-parameters.
        use_progress_bar (bool): Use tqdm progress bar (can be disabled when logging).
        accelerator: PL trainer accelerator
        devices: PL devices to use
    """
    check_and_log_hp(['max_epoch'], hyper_params)

    # TODO pl Trainer does not use the kwarg resume_from_checkpoint now - check about resume training works now
    # resume_from_checkpoint = handle_previous_models(output, last_model_path, best_model_path)

    callbacks_dict = create_all_callbacks(hyper_params, output, verbose=use_progress_bar)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=output,
        default_hp_metric=False,
        version=0,  # Necessary to resume tensorboard logging
    )

    trainer = pl.Trainer(
        callbacks=list(callbacks_dict.values()),
        max_epochs=hyper_params['max_epoch'],
        # resume_from_checkpoint=resume_from_checkpoint,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
    )

    # Using the keyword ckpt_path="last" tells the trainer to resume from the last
    # checkpoint, or to start from scratch if none exist.
    trainer.fit(model, datamodule=datamodule, ckpt_path='last')

    # By convention, it is assumed that the metric to be reported is the early stopping metric.
    if 'early_stopping' in callbacks_dict:
        early_stopping = callbacks_dict['early_stopping']
        best_value = float(early_stopping.best_score.cpu().numpy())

        metric_name, mode = get_optimized_metric_name_and_mode(hyper_params)
        logger.log_hyperparams(hyper_params, metrics={metric_name: best_value})
        metric_result = MetricResult(report=True, metric_name=metric_name, mode=mode, metric_value=best_value)
    else:
        metric_result = MetricResult(report=False)

    return metric_result


if __name__ == '__main__':
    main()

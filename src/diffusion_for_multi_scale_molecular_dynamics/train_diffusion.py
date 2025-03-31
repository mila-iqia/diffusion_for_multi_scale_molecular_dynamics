"""Entry point to train a diffusion model."""

import argparse
import logging
import os
import shutil
import typing

import pytorch_lightning
import pytorch_lightning as pl
import yaml

from diffusion_for_multi_scale_molecular_dynamics.callbacks.callback_loader import \
    create_all_callbacks
from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.instantiate_data_module import \
    load_data_module
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.loggers.logger_loader import \
    create_all_loggers
from diffusion_for_multi_scale_molecular_dynamics.models.instantiate_diffusion_model import \
    load_diffusion_model
from diffusion_for_multi_scale_molecular_dynamics.utils.hp_utils import \
    check_and_log_hp
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import (
    configure_logging, log_exp_details)
from diffusion_for_multi_scale_molecular_dynamics.utils.main_utils import (
    MetricResult, get_crash_metric_result, get_optimized_metric_name_and_mode,
    load_and_backup_hyperparameters, report_to_orion_if_on)

logger = logging.getLogger(__name__)


def main(args: typing.Optional[typing.Any] = None):
    """Create and train a diffusion model: main entry point of the program.

    Note:
        This main.py file is meant to be called using the cli,
        see the `examples/local/run_diffusion.sh` file to see how to use it.
    """
    parser = argparse.ArgumentParser()
    # __TODO__ check you need all the following CLI parameters
    parser.add_argument(
        "--config",
        help="config file with generic hyper-parameters,  such as optimizer, "
        "batch_size, ... -  in yaml format",
    )
    parser.add_argument("--data",
                        help="path to a LAMMPS data set. REQUIRED if the data source is 'LAMMPS'.",
                        default=None,
                        required=False)
    parser.add_argument(
        "--processed_datadir",
        help="path to the processed data directory. REQUIRED if the data source is 'LAMMPS'.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--dataset_working_dir",
        help="path to the Datasets working directory. Only relevant if the data source is 'LAMMPS'. Defaults to None",
        default=None,
    )
    parser.add_argument(
        "--output", help="path to outputs - will store files here", required=True
    )
    parser.add_argument(
        "--disable-progressbar",
        action="store_true",
        help="will disable the progressbar while going over the mini-batch",
    )
    parser.add_argument(
        "--start-from-scratch",
        action="store_true",
        help="will not load any existing saved model - even if present",
    )
    parser.add_argument(
        "--accelerator",
        help="PL trainer accelerator. Defaults to auto.",
        default="auto",
    )
    parser.add_argument(
        "--devices", default=1, help="pytorch-lightning devices kwarg. Defaults to 1."
    )
    args = parser.parse_args(args)

    if os.path.exists(args.output) and args.start_from_scratch:
        first_logging_message = "Previous experiment found: starting from scratch, removing any previous experiments."
        shutil.rmtree(args.output)
        os.makedirs(args.output)
    elif os.path.exists(args.output):
        first_logging_message = "Previous experiment found: resuming from checkpoint"
    else:
        first_logging_message = "NO previous experiment found: starting from scratch"
        os.makedirs(args.output)

    # Very opinionated logger, which writes to the output folder.
    configure_logging(experiment_dir=args.output)
    logger.info(first_logging_message)
    log_exp_details(os.path.realpath(__file__), args)

    output_dir = args.output

    hyper_params = load_and_backup_hyperparameters(
        config_file_path=args.config, output_directory=output_dir
    )

    logger.info(
        "Input hyper-parameters:\n"
        + yaml.dump(hyper_params, allow_unicode=True, default_flow_style=False)
    )

    run(args, output_dir, hyper_params)


def run(args: argparse.Namespace, output_dir, hyper_params):
    """Create and run the dataloaders, training loops, etc.

    Args:
        args (object): arguments passed from the cli
        output_dir (str): path to output folder
        hyper_params (dict): hyper parameters from the config file
    """
    # __TODO__ change the hparam that are used from the training algorithm
    # (and NOT the model - these will be specified in the model itself)
    if hyper_params["seed"] is not None:
        pytorch_lightning.seed_everything(hyper_params["seed"])

    ElementTypes.validate_elements(hyper_params["elements"])

    datamodule = load_data_module(hyper_params, args)

    model = load_diffusion_model(hyper_params)

    try:
        metric_result = train(
            model=model,
            datamodule=datamodule,
            output=output_dir,
            hyper_params=hyper_params,
            use_progress_bar=not args.disable_progressbar,
            accelerator=args.accelerator,
            devices=args.devices,
        )
        run_time_error = None
    except RuntimeError as err:
        run_time_error = err
        logger.error(err)
        metric_result = get_crash_metric_result(hyper_params)

    # clean up the data cache to save disk space
    datamodule.clean_up()

    report_to_orion_if_on(metric_result, run_time_error)


def train(
    model,
    datamodule,
    output: str,
    hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    use_progress_bar: bool,
    accelerator=None,
    devices=None,
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
    check_and_log_hp(["max_epoch"], hyper_params)

    callbacks_dict = create_all_callbacks(
        hyper_params, output, verbose=use_progress_bar
    )
    pl_loggers = create_all_loggers(hyper_params, output)
    for pl_logger in pl_loggers:
        pl_logger.log_hyperparams(hyper_params)

    trainer = pl.Trainer(
        callbacks=list(callbacks_dict.values()),
        max_epochs=hyper_params["max_epoch"],
        log_every_n_steps=hyper_params.get("log_every_n_steps", None),
        fast_dev_run=hyper_params.get("fast_dev_run", False),
        accelerator=accelerator,
        devices=devices,
        logger=pl_loggers,
        gradient_clip_val=hyper_params.get("gradient_clipping", 0),
        accumulate_grad_batches=hyper_params.get("accumulate_grad_batches", 1),
    )

    # Using the keyword ckpt_path="last" tells the trainer to resume from the last
    # checkpoint, or to start from scratch if none exists.
    trainer.fit(model, datamodule=datamodule, ckpt_path="last")

    # By convention, it is assumed that the metric to be reported is the early stopping metric.
    if "early_stopping" in callbacks_dict:
        early_stopping = callbacks_dict["early_stopping"]
        best_value = float(early_stopping.best_score.cpu().numpy())

        metric_name, mode = get_optimized_metric_name_and_mode(hyper_params)
        for pl_logger in pl_loggers:
            pl_logger.log_metrics({f"best_{metric_name}": best_value})

        metric_result = MetricResult(
            report=True, metric_name=metric_name, mode=mode, metric_value=best_value
        )
    else:
        metric_result = MetricResult(report=False)

    return metric_result


if __name__ == "__main__":
    # Uncomment the following in order to use Pycharm's Remote Debugging server, which allows to
    # launch python commands through a bash script (and through Orion!). VERY useful for debugging.
    # This requires a professional edition of Pycharm and installing the pydevd_pycharm package with pip.
    # The debug server stopped working in 2024.3. There is a workaround. See:
    #   https://www.reddit.com/r/pycharm/comments/1gs1lgk/python_debug_server_issues/
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=56636, stdoutToServer=True, stderrToServer=True)
    main()

import os
import uuid
from typing import Any, AnyStr, Dict, List, Union

import orion
import yaml
from pytorch_lightning.loggers import (CometLogger, CSVLogger, Logger,
                                       TensorBoardLogger)


def get_run_name(hyper_params: Dict[AnyStr, Any]) -> str:
    """Get run name.

    Either create or retrieve the run name for the current experiment.

    Args:
        hyper_params : Dictionary of configuration parameters

    Returns:
        run_name: the name of the run.
    """
    default_run_name = uuid.uuid4().hex  # will be used in case we've got no other ID...
    if orion.client.cli.IS_ORION_ON and "run_name" not in hyper_params:
        run_name = str(os.getenv("ORION_TRIAL_ID", default=default_run_name))
    else:
        run_name = str(hyper_params.get("run_name", default_run_name))
    return run_name


def create_all_loggers(hyper_params: Dict[AnyStr, Any], output_directory: str) -> List[Logger]:
    """Create all loggers.

    This method instantiates all machine learning loggers defined in `hyper_params`.

    Args:
        hyper_params : configuration parameters.
        output_directory: path to where outputs are to be written.
        verbose: if relevant, should the callback produce verbose output.

    Returns:
        all_loggers : a list of loggers to pass to the Trainer.
    """
    all_loggers = []

    experiment_name = hyper_params.get("exp_name", "DEFAULT")
    run_name = get_run_name(hyper_params)

    full_run_name = f"{experiment_name}/{run_name}"

    for logger_name in hyper_params.get('logging', []):

        match logger_name:
            case "csv":
                logger = CSVLogger(save_dir=output_directory,
                                   name=full_run_name)
            case "tensorboard":
                logger = TensorBoardLogger(save_dir=output_directory,
                                           default_hp_metric=False,
                                           name=full_run_name,
                                           version=0,  # Necessary to resume tensorboard logging
                                           )
            case "comet":
                # The comet logger will read the API key from ~/.comet.logger.

                # Pytorch Lightning's wrapper around Comet, the CometLogger class, makes it impossible
                # to define our own 'experiment_key'; it defers to Comet for the creation of such a key when
                # starting a new experiment (see pytorch_lightning.loggers.comet.CometLogger.experiment).
                # To avoid creating our own CometLogger (which would be time-consuming and costly to maintain),
                # Here we adopt the following strategy:
                #   - when creating a NEW experiment, write the experiment key to a yaml file to the output directory;
                #   - when resuming an experiment, read the experiment key from the yaml file.
                #   --> The presence or absence of this yaml file will serve as a signal of whether are starting
                #       from scratch or resuming.
                experiment_key = read_and_validate_comet_experiment_key(full_run_name=full_run_name,
                                                                        output_directory=output_directory)

                # if 'experiment_key' is None, a new Comet Experiment will be created. If it is not None,
                # an ExistingExperiment will be created in order to resume logging.
                logger = CometLogger(project_name='institut-courtois',      # hardcoding the project.
                                     save_dir=output_directory,
                                     experiment_key=experiment_key,
                                     experiment_name=experiment_name)
                if experiment_key is None:
                    experiment_key = logger.version
                    write_comet_experiment_key(experiment_key=experiment_key,
                                               full_run_name=full_run_name,
                                               output_directory=output_directory)

            case other:
                raise ValueError(f"Logger {other} is not implemented. Review input")

        all_loggers.append(logger)

    return all_loggers


def write_comet_experiment_key(experiment_key: str, full_run_name: str, output_directory: str):
    """Write comet experiment key.

    Args:
        experiment_key : the Comet experiment identifier
        full_run_name : the complete run name for the experiment (experiment_name / run_name)
        output_directory : directory where the results are written.

    Returns:
        No returns.
    """
    data = {full_run_name: experiment_key}
    with open(os.path.join(output_directory, "comet_experiment_key.yaml"), 'w') as fd:
        yaml.dump(data, fd)


def read_and_validate_comet_experiment_key(full_run_name: str, output_directory: str) -> Union[str, None]:
    """Read and validate Comet's experiment key.

    Args:
        full_run_name : the complete run name for the experiment (experiment_name / run_name)
        output_directory : directory where the results are written.

    Returns:
        experiment_key : the Comet experiment identifier, if it exists, or else None.
    """
    experiment_key = None
    key_file = os.path.join(output_directory, "comet_experiment_key.yaml")
    if os.path.isfile(key_file):
        with open(key_file) as fd:
            data = yaml.load(fd, Loader=yaml.FullLoader)
            assert full_run_name in data, (f"The experiment full name is {full_run_name}; this is not the full name "
                                           f"that was used to initially generate this experiment. "
                                           f"Something is inconsistent and requires a manual fix.")
            experiment_key = data[full_run_name]

    return experiment_key

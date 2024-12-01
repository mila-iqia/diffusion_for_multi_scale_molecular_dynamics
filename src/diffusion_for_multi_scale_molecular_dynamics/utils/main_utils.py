import logging
import os
from dataclasses import dataclass
from typing import Any, AnyStr, Dict, Tuple, Union

import deepdiff
import numpy as np
import orion.client
import yaml

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MetricResult:
    """Metric result class that is self documenting."""

    report: bool = False  # is there something to report
    metric_name: Union[str, None] = (
        None  # default to None, if there is nothing to report
    )
    mode: Union[str, None]  # default to None, if there is nothing to report
    metric_value: float = np.NaN  # default to NaN, if there is nothing to report


def get_optimized_metric_name_and_mode(
    hyper_params: Dict[AnyStr, Any]
) -> Tuple[Union[str, None], Union[str, None]]:
    """Get optimized metric name and mode.

    Args:
        hyper_params : Dict containing hyper-parameters.

    Returns:
        metric_name, metric_mode: the name and mode (min or max) for the metric to be optimized.
    """
    # By convention, it is assumed that the metric to be reported is the early stopping metric.
    if "early_stopping" in hyper_params:
        early_stopping_params = hyper_params["early_stopping"]
        return early_stopping_params["metric"], early_stopping_params["mode"]

    else:
        return None, None


def get_crash_metric_result(hyper_params: Dict[AnyStr, Any]) -> MetricResult:
    """Get crash metric result.

    Produce a MetricResult object that is appropriate for reporting when there is a run time error during training.

    Args:
        hyper_params : Dict containing hyper-parameters.

    Returns:
        crash_metric_result: appropriate result when there is a code crash.
    """
    metric_name, mode = get_optimized_metric_name_and_mode(hyper_params)

    if metric_name is None:
        return MetricResult(
            report=False, metric_name=None, mode=None, metric_value=np.NaN
        )
    else:
        return MetricResult(
            report=True, metric_name=metric_name, mode=mode, metric_value=np.NaN
        )


def get_name_and_sign_of_orion_optimization_objective(
    metric_name: str, mode: str
) -> Tuple[str, int]:
    """Names and signs.

    The Orion optimizer seeks to minimize an objective. Some metrics must be maximized,
    and others must be minimized. This function returns what is needed to align with Orion.

    Args:
        metric_name: name of the early stop metric, as passed in the input config file.
        mode: optimization mode for the metric ('min' or 'max')

    Returns:
        optimization_objective_name: A proper name for what Orion will be optimizing
        optimization_sign: premultiplicative factor (+/- 1) to make sure Orion tries to minimize an objective.
    """
    if mode == "max":
        # The metric must be maximized. Correspondingly, Orion will minimize minus x metric.
        optimization_objective_name = f"minus_{metric_name}"
        optimization_sign = -1
    elif mode == "min":
        # The metric must be minimized; this is already aligned with what Orion will optimize.
        optimization_objective_name = metric_name
        optimization_sign = 1
    else:
        raise ValueError("The mode for this early_stopping_metric is unknown")
    return optimization_objective_name, optimization_sign


def report_to_orion_if_on(
    metric_result: MetricResult, run_time_error: Union[None, RuntimeError]
):
    """Report to Orion if on.

    This function manages how to report the metric to Orion. If Orion is not turned on, or if there
    is nothing to report, this method has no effect.

    Args:
        metric_result: self-documenting result to report.
        run_time_error : error if the training failed, None if all went well.

    Returns:
        No returns.
    """
    if not metric_result.report:
        # There is nothing to report.
        return

    if orion.client.cli.IS_ORION_ON:
        optimization_objective_name, optimization_sign = (
            get_name_and_sign_of_orion_optimization_objective(
                metric_result.metric_name, metric_result.mode
            )
        )

        report_value = optimization_sign * metric_result.metric_value

        if run_time_error is None:
            logger.info("Reporting Results to ORION...")
            results = dict(
                name=optimization_objective_name, type="objective", value=report_value
            )
            orion.client.report_results([results])

            logger.info(" Done reporting Results to ORION.")
        elif "CUDA out of memory" in str(run_time_error):
            logger.error(
                "model was out of memory - reporting a bad trial so "
                "that Orion avoids models that are too large"
            )
            orion.client.report_bad_trial(name=optimization_objective_name)
        else:
            logger.error(f"Run time error : {run_time_error}- interrupting Orion trial")
            orion.client.interrupt_trial()


def load_and_backup_hyperparameters(
    config_file_path: Union[str, None], output_directory: str
) -> Dict[str, Any]:
    """Load and process hyperparameters.

    If a configuration file is provided, this method reads in the hyperparameters. It either makes a copy of the
    configuration parameters to the output_directory, or validates that the hyperparameters are the same
    as those in the config file already in the output directory (this may happen in the case of a job being resumed).

    If Orion is being used, no copy / validation will take place since Orion already manages copies of configuration
    files.

    Args:
        config_file_path : path to config file, if there is one.
        output_directory : directory where results are output.

    Returns:
        hyperparameters: all the read hyperparameters. Returns empty dictionary if there is no config file.
    """
    hyper_params = _get_hyperparameters(config_file_path)

    if orion.client.cli.IS_ORION_ON:
        logging.info(
            "The Orion client is ON: Orion will manage configuration file copies."
        )
    else:
        config_backup_path = os.path.join(output_directory, "config_backup.yaml")
        _create_or_validate_backup_configuration(config_backup_path, hyper_params)

    return hyper_params


def _create_or_validate_backup_configuration(
    config_backup_path: str, hyper_params: Dict[str, Any]
) -> None:
    """Create or validate a backup of the hyperparameters."""
    if os.path.exists(config_backup_path):
        logging.info(
            f"The backup configuration file {config_backup_path} already exists. "
            f"Validating hyperparameters are identical..."
        )

        with open(config_backup_path, "r") as stream:
            logging.info(
                f"Reading backup hyperparameters from file {config_backup_path}"
            )
            backup_hyper_params = yaml.load(stream, Loader=yaml.FullLoader)

        hp_differences = deepdiff.DeepDiff(backup_hyper_params, hyper_params)
        assert hp_differences == {}, (
            f"Incompatible backup configuration file already present in output directory! "
            f"The configuration difference is {hp_differences}. Manual clean up is needed."
        )

    else:
        logging.info(
            f"Writing a copy of the configuration file to backup configuration file {config_backup_path}."
        )
        with open(config_backup_path, "w") as steam:
            yaml.dump(hyper_params, steam)


def _get_hyperparameters(config_file_path: Union[str, None]) -> Dict[str, Any]:
    """Get the hyperparameters."""
    if config_file_path is None:
        logging.info(
            "No configuration file was provided. The hyperparameters are set to an empty dictionary."
        )
        hyper_params = dict()
    else:
        logging.info(f"Reading in hyperparameters from file {config_file_path}")
        with open(config_file_path, "r") as stream:
            hyper_params = yaml.load(stream, Loader=yaml.FullLoader)
    return hyper_params

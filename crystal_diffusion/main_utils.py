import logging
from dataclasses import dataclass
from typing import Any, AnyStr, Dict, Tuple, Union

import numpy as np
import orion.client

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MetricResult:
    """Metric result class that is self documenting."""
    report: bool = False  # is there something to report
    metric_name: Union[str, None] = None  # default to None, if there is nothing to report
    mode: Union[str, None]  # default to None, if there is nothing to report
    metric_value: float = np.NaN  # default to NaN, if there is nothing to report


def get_optimized_metric_name_and_mode(hyper_params: Dict[AnyStr, Any]) -> Tuple[Union[str, None], Union[str, None]]:
    """Get optimized metric name and mode.

    Args:
        hyper_params : Dict containing hyper-parameters.

    Returns:
        metric_name, metric_mode: the name and mode (min or max) for the metric to be optimized.
    """
    # By convention, it is assumed that the metric to be reported is the early stopping metric.
    if 'early_stopping' in hyper_params:
        early_stopping_params = hyper_params['early_stopping']
        return early_stopping_params['metric'], early_stopping_params['mode']

    else:
        return None, None


def get_crash_metric_result(hyper_params: Dict[AnyStr, Any]) -> MetricResult:
    """Get crash metric result.

    Produce a MetricResult object that is appropriate for reporting when there is a run time error during training..

    Args:
        hyper_params : Dict containing hyper-parameters.

    Returns:
        crash_metric_result: appropriate result when there is a code crash.
    """
    metric_name, mode = get_optimized_metric_name_and_mode(hyper_params)

    if metric_name is None:
        return MetricResult(report=False, metric_name=None, mode=None, metric_value=np.NaN)
    else:
        return MetricResult(report=True, metric_name=metric_name, mode=mode, metric_value=np.NaN)


def get_name_and_sign_of_orion_optimization_objective(metric_name: str, mode: str) -> Tuple[str, int]:
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


def report_to_orion_if_on(metric_result: MetricResult, run_time_error: Union[None, RuntimeError]):
    """Report to orion if on.

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
            get_name_and_sign_of_orion_optimization_objective(metric_result.metric_name, metric_result.mode))

        report_value = optimization_sign * metric_result.metric_value

        if run_time_error is None:
            logger.info("Reporting Results to ORION...")
            results = dict(name=optimization_objective_name, type="objective", value=report_value)
            orion.client.report_results([results])

            logger.info(" Done reporting Results to ORION.")
        elif 'CUDA out of memory' in str(run_time_error):
            logger.error('model was out of memory - reporting a bad trial so '
                         'that Orion avoids models that are too large')
            orion.client.report_bad_trial(name=optimization_objective_name)
        else:
            logger.error(f"Run time error : {run_time_error}- interrupting Orion trial")
            orion.client.interrupt_trial()

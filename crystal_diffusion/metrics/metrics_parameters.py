from dataclasses import dataclass
from typing import Any, AnyStr, Dict, Union


@dataclass(kw_only=True)
class MetricsParameters:
    """Metrics parameters.

    This dataclass describes which metrics should be computed.
    """
    fokker_planck: bool = False


def load_metrics_parameters(hyper_params: Dict[AnyStr, Any]) -> Union[MetricsParameters, None]:
    """Load metrics parameters.

    Extract the needed information from the configuration dictionary.

    Args:
        hyper_params: dictionary of hyperparameters loaded from a config file

    Returns:
        metrics_parameters: the relevant configuration object.
    """
    if 'metrics' not in hyper_params:
        return None

    return MetricsParameters(**hyper_params['metrics'])

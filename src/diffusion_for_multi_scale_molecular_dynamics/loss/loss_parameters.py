from dataclasses import dataclass
from typing import Any, Dict

from diffusion_for_multi_scale_molecular_dynamics.utils.configuration_parsing import \
    create_parameters_from_configuration_dictionary


@dataclass(kw_only=True)
class LossParameters:
    """Specific Hyper-parameters for the loss function."""

    coordinates_algorithm: str
    atom_types_ce_weight: float = 0.001  # default value in google D3PM repo
    atom_types_eps: float = 1e-8  # avoid divisions by zero
    # https://github.com/google-research/google-research/blob/master/d3pm/images/config.py


@dataclass(kw_only=True)
class MSELossParameters(LossParameters):
    """Specific Hyper-parameters for the MSE loss function."""

    coordinates_algorithm: str = "mse"


@dataclass(kw_only=True)
class WeightedMSELossParameters(LossParameters):
    """Specific Hyper-parameters for the weighted MSE loss function."""

    coordinates_algorithm: str = "weighted_mse"
    # The default values are chosen to lead to a flat loss curve vs. sigma, based on preliminary experiments.
    # These parameters have no effect if the algorithm is 'mse'.
    # The default parameters are chosen such that weights(sigma=0.5) \sim 10^3
    sigma0: float = 0.2
    exponent: float = 23.0259  # ~ 10 ln(10)


def create_loss_parameters(model_dictionary: Dict[str, Any]) -> LossParameters:
    """Create loss parameters.

    Extract the relevant information from the general configuration dictionary.

    Args:
        model_dictionary : model configuration dictionary.

    Returns:
        loss_parameters: the loss parameters.
    """
    default_dict = dict(algorithm="mse")
    loss_config_dictionary = model_dictionary.get("loss", default_dict)

    loss_parameters = create_parameters_from_configuration_dictionary(
        configuration=loss_config_dictionary,
        identifier="coordinates_algorithm",
        options=LOSS_PARAMETERS_BY_ALGO,
    )
    return loss_parameters


LOSS_PARAMETERS_BY_ALGO = dict(
    mse=MSELossParameters, weighted_mse=WeightedMSELossParameters
)

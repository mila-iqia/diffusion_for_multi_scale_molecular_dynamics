from dataclasses import dataclass
from typing import Any, Dict

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.configuration_parsing import \
    create_parameters_from_configuration_dictionary


@dataclass(kw_only=True)
class LossParameters:
    """Hyper-parameters for the loss function for a single modality (A, X xor L)."""

    lambda_weight: float = 1.0
    algorithm: str


@dataclass(kw_only=True)
class MSELossParameters(LossParameters):
    """Specific Hyper-parameters for the MSE loss function."""

    algorithm: str = "mse"


@dataclass(kw_only=True)
class WeightedMSELossParameters(LossParameters):
    """Specific Hyper-parameters for the weighted MSE loss function."""

    algorithm: str = "weighted_mse"
    # The default values are chosen to lead to a flat loss curve vs. sigma, based on preliminary experiments.
    # These parameters have no effect if the algorithm is 'mse'.
    # The default parameters are chosen such that weights(sigma=0.5) \sim 10^3
    sigma0: float = 0.2
    exponent: float = 23.0259  # ~ 10 ln(10)


@dataclass(kw_only=True)
class AtomTypeLossParameters(LossParameters):
    algorithm: str = "d3pm"
    ce_weight: float = 0.001  # default value in google D3PM repo
    eps: float = 1e-8  # avoid divisions by zero
    # https://github.com/google-research/google-research/blob/master/d3pm/images/config.py


def create_loss_parameters(model_dictionary: Dict[str, Any]) -> AXL:
    """Create loss parameters.

    Extract the relevant information from the general configuration dictionary.

    Args:
        model_dictionary : model configuration dictionary.

    Returns:
        loss_parameters: the loss parameters in an AXL named tuple.
    """
    default_mse_dict = dict(algorithm="mse")
    default_d3pm_dict = dict(algorithm="d3pm")
    default_axl_dict = dict(
        coordinates=default_mse_dict,
        atom_types=default_d3pm_dict,
        lattice_parameters=default_mse_dict,
    )
    loss_config_dictionary = model_dictionary.get("loss", default_axl_dict)

    loss_parameters = {}
    for var in ["coordinates", "atom_types", "lattice_parameters"]:
        default_params = default_d3pm_dict if var == "atom_types" else default_mse_dict
        loss_parameters[var] = create_parameters_from_configuration_dictionary(
            configuration=loss_config_dictionary.get(var, default_params),
            identifier="algorithm",
            options=LOSS_PARAMETERS_BY_ALGO,
        )
    loss_parameters = AXL(
        A=loss_parameters["atom_types"],
        X=loss_parameters["coordinates"],
        L=loss_parameters["lattice_parameters"],
    )

    return loss_parameters


LOSS_PARAMETERS_BY_ALGO = dict(
    mse=MSELossParameters,
    weighted_mse=WeightedMSELossParameters,
    d3pm=AtomTypeLossParameters,
)

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from crystal_diffusion.utils.configuration_parsing import \
    create_parameters_from_configuration_dictionary
from torch import optim

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class OptimizerParameters:
    """Parameters for the optimizer."""
    name: str
    learning_rate: float
    weight_decay: float = 0.0


OPTIMIZERS_BY_NAME = dict(adam=optim.Adam, adamw=optim.AdamW)
OPTIMIZER_PARAMETERS_BY_NAME = dict(adam=OptimizerParameters, adamw=OptimizerParameters)


def create_optimizer_parameters(optimizer_configuration_dictionary: Dict[str, Any]) -> OptimizerParameters:
    """Create Optimizer Parameters.

    Args:
        optimizer_configuration_dictionary : optimizer-related parameters.

    Returns:
        optimizer_parameters: a Dataclass with the optimizer parameters.
    """
    optimizer_parameters = (
        create_parameters_from_configuration_dictionary(configuration=optimizer_configuration_dictionary,
                                                        identifier="name",
                                                        options=OPTIMIZER_PARAMETERS_BY_NAME))
    return optimizer_parameters


def load_optimizer(optimizer_parameters: OptimizerParameters, model: torch.nn.Module) -> optim.Optimizer:
    """Instantiate the optimizer.

    Args:
        optimizer_parameters: hyperparameters defining the optimizer
        model : A neural network model.

    Returns:
        optimizer : The optimizer for the given model
    """
    assert optimizer_parameters.name in OPTIMIZERS_BY_NAME.keys(), \
        f"Optimizer '{optimizer_parameters.name}' is not defined."
    optimizer_class = OPTIMIZERS_BY_NAME[optimizer_parameters.name]

    # Adapt the input configurations to the optimizer constructor expectations.
    parameters_dict = asdict(optimizer_parameters)
    parameters_dict.pop('name')
    parameters_dict['lr'] = parameters_dict.pop('learning_rate')

    return optimizer_class(model.parameters(), **parameters_dict)

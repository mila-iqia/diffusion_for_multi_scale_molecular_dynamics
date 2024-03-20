import logging
from dataclasses import dataclass
from enum import Enum

import torch
from torch import optim

logger = logging.getLogger(__name__)


class ValidOptimizerNames(Enum):
    """Valid optimizer names."""
    adam = "adam"
    sgd = "sgd"


@dataclass(kw_only=True)
class OptimizerParameters:
    """Parameters for the optimizer."""
    name: ValidOptimizerNames
    learning_rate: float


def load_optimizer(hyper_params: OptimizerParameters, model: torch.nn.Module) -> optim.Optimizer:
    """Instantiate the optimizer.

    Args:
        hyper_params : hyperparameters defining the optimizer
        model : A neural network model.

    Returns:
        optimizer : The optimizer for the given model
    """
    match hyper_params.name:
        case ValidOptimizerNames.adam:
            optimizer = optim.Adam(model.parameters(), lr=hyper_params.learning_rate)
        case ValidOptimizerNames.sgd:
            optimizer = optim.SGD(model.parameters(), lr=hyper_params.learning_rate)
        case _:
            raise ValueError(f"optimizer {hyper_params.name} not supported")
    return optimizer

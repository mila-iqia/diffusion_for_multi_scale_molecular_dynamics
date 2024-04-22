import logging
from dataclasses import dataclass
from enum import Enum

import torch
from torch import optim

logger = logging.getLogger(__name__)


class ValidOptimizerName(Enum):
    """Valid optimizer names."""
    adam = "adam"
    adamw = "adamw"


@dataclass(kw_only=True)
class OptimizerParameters:
    """Parameters for the optimizer."""
    name: ValidOptimizerName
    learning_rate: float
    weight_decay: float = 0.0


def load_optimizer(hyper_params: OptimizerParameters, model: torch.nn.Module) -> optim.Optimizer:
    """Instantiate the optimizer.

    Args:
        hyper_params : hyperparameters defining the optimizer
        model : A neural network model.

    Returns:
        optimizer : The optimizer for the given model
    """
    parameters_dict = dict(lr=hyper_params.learning_rate, weight_decay=hyper_params.weight_decay)
    match hyper_params.name:
        case ValidOptimizerName.adam:
            optimizer = optim.Adam(model.parameters(), **parameters_dict)
        case ValidOptimizerName.adamw:
            optimizer = optim.AdamW(model.parameters(), **parameters_dict)
        case _:
            raise ValueError(f"optimizer {hyper_params.name} not supported")
    return optimizer

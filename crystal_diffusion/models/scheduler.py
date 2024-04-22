import dataclasses
from dataclasses import dataclass
from enum import Enum

from torch import optim


class ValidSchedulerName(Enum):
    """Valid scheduler names."""
    reduce_lr_on_plateau = "ReduceLROnPlateau"
    cosine_annealing_lr = "CosineAnnealingLR"


@dataclass(kw_only=True)
class SchedulerParameters:
    """Base data class for scheduler parameters."""
    name: ValidSchedulerName


@dataclass(kw_only=True)
class ReduceLROnPlateauSchedulerParameters(SchedulerParameters):
    """Parameters for the reduce LR on plateau scheduler."""
    factor: float = 0.1
    patience: int = 10


@dataclass(kw_only=True)
class CosineAnnealingLRSchedulerParameters(SchedulerParameters):
    """Parameters for the reduce LR on plateau scheduler."""
    T_max: int
    eta_min: float = 0.0


def load_scheduler(hyper_params: SchedulerParameters, optimizer: optim.Optimizer) -> optim.lr_scheduler:
    """Instantiate the Scheduler.

    Args:
        hyper_params : hyperparameters defining the scheduler
        optimizer: the optimizer to be scheduled.

    Returns:
        scheduler : the configured scheduler.
    """
    parameters_dict = dataclasses.asdict(hyper_params)
    parameters_dict.pop('name')

    match hyper_params.name:
        case ValidSchedulerName.reduce_lr_on_plateau:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **parameters_dict)
        case ValidSchedulerName.cosine_annealing_lr:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **parameters_dict)
        case _:
            raise ValueError(f"scheduler {hyper_params.name} not supported")

    return scheduler

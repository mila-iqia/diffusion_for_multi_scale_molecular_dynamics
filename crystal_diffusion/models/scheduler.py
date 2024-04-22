import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import AnyStr, Dict, Union

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
    # The scheduler must be told what to monitor. This is not actually a parameter for the scheduler itself,
    # but  must be provided when configuring optimizers and schedulers.
    monitor: str = "validation_epoch_loss"


@dataclass(kw_only=True)
class CosineAnnealingLRSchedulerParameters(SchedulerParameters):
    """Parameters for the reduce LR on plateau scheduler."""
    T_max: int
    eta_min: float = 0.0


def load_scheduler_dictionary(hyper_params: SchedulerParameters,
                              optimizer: optim.Optimizer) -> Dict[AnyStr, Union[optim.lr_scheduler, AnyStr]]:
    """Instantiate the Scheduler.

    Args:
        hyper_params : hyperparameters defining the scheduler
        optimizer: the optimizer to be scheduled.

    Returns:
        scheduler_dict : A dictionary containing the configured scheduler, as well as potentially other needed info.
    """
    scheduler_dict = dict()
    parameters_dict = dataclasses.asdict(hyper_params)
    parameters_dict.pop('name')

    match hyper_params.name:
        case ValidSchedulerName.reduce_lr_on_plateau:
            scheduler_dict["monitor"] = parameters_dict.pop('monitor')
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **parameters_dict)
        case ValidSchedulerName.cosine_annealing_lr:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **parameters_dict)
        case _:
            raise ValueError(f"scheduler {hyper_params.name} not supported")

    scheduler_dict['lr_scheduler'] = scheduler
    return scheduler_dict

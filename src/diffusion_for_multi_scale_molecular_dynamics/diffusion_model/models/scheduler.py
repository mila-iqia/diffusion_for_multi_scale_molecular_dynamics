from dataclasses import asdict, dataclass
from typing import Any, AnyStr, Dict, Union

from torch import optim

from diffusion_for_multi_scale_molecular_dynamics.utils.configuration_parsing import \
    create_parameters_from_configuration_dictionary


@dataclass(kw_only=True)
class SchedulerParameters:
    """Base data class for scheduler parameters."""

    name: str


@dataclass(kw_only=True)
class ReduceLROnPlateauSchedulerParameters(SchedulerParameters):
    """Parameters for the reduce LR on plateau scheduler."""

    name: str = "ReduceLROnPlateau"
    factor: float = 0.1
    patience: int = 10
    # The scheduler must be told what to monitor. This is not actually a parameter for the scheduler itself,
    # but  must be provided when configuring optimizers and schedulers.
    monitor: str = "validation_epoch_loss"


@dataclass(kw_only=True)
class CosineAnnealingLRSchedulerParameters(SchedulerParameters):
    """Parameters for the reduce LR on plateau scheduler."""

    name: str = "CosineAnnealingLR"
    T_max: int
    eta_min: float = 0.0


SCHEDULER_PARAMETERS_BY_NAME = dict(
    CosineAnnealingLR=CosineAnnealingLRSchedulerParameters,
    ReduceLROnPlateau=ReduceLROnPlateauSchedulerParameters,
)

SCHEDULERS_BY_NAME = dict(
    CosineAnnealingLR=optim.lr_scheduler.CosineAnnealingLR,
    ReduceLROnPlateau=optim.lr_scheduler.ReduceLROnPlateau,
)


def create_scheduler_parameters(
    hyper_params: Dict[str, Any]
) -> Union[SchedulerParameters, None]:
    """Create scheduler parameters.

    Extract the relevant information from the general configuration dictionary.

    Args:
        hyper_params : configuration dictionary.

    Returns:
        scheduler_parameters: the scheduler parameters.
    """
    if "scheduler" not in hyper_params:
        return None

    scheduler_configuration_dict = dict(hyper_params["scheduler"])

    scheduler_parameters = create_parameters_from_configuration_dictionary(
        configuration=scheduler_configuration_dict,
        identifier="name",
        options=SCHEDULER_PARAMETERS_BY_NAME,
    )
    return scheduler_parameters


def load_scheduler_dictionary(
    scheduler_parameters: SchedulerParameters, optimizer: optim.Optimizer
) -> Dict[AnyStr, Any]:
    """Instantiate the Scheduler.

    Args:
        scheduler_parameters: hyperparameters defining the scheduler
        optimizer: the optimizer to be scheduled.

    Returns:
        scheduler_dict : A dictionary containing the configured scheduler, as well as potentially other needed info.
    """
    name = scheduler_parameters.name
    assert (
        name in SCHEDULERS_BY_NAME.keys()
    ), f"Scheduler '{name}' is not implemented. Possible choices are {SCHEDULERS_BY_NAME.keys()}"

    scheduler_dict = dict()

    scheduler_class = SCHEDULERS_BY_NAME[name]

    scheduler_parameters_dict = asdict(scheduler_parameters)
    scheduler_parameters_dict.pop("name")

    if "monitor" in scheduler_parameters_dict:
        scheduler_dict["monitor"] = scheduler_parameters_dict.pop("monitor")

    scheduler_dict["lr_scheduler"] = scheduler_class(
        optimizer, **scheduler_parameters_dict
    )

    return scheduler_dict

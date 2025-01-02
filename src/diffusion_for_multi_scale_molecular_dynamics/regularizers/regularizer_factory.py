from typing import Any, AnyStr, Dict

from diffusion_for_multi_scale_molecular_dynamics.regularizers.fokker_planck_regularizer import (
    FokkerPlanckRegularizer, FokkerPlanckRegularizerParameters)
from diffusion_for_multi_scale_molecular_dynamics.regularizers.regularizer import (
    Regularizer, RegularizerParameters)

REGULARIZERS_BY_TYPE = dict(
    fokker_planck=FokkerPlanckRegularizer
)

REGULARIZER_PARAMETERS_BY_TYPE = dict(
    fokker_planck=FokkerPlanckRegularizerParameters,
)


def create_regularizer(regularizer_parameters: RegularizerParameters) -> Regularizer:
    """Create regularizer."""
    type = regularizer_parameters.type
    assert (
        type in REGULARIZERS_BY_TYPE.keys()
    ), f"Regularizer type {type} is not implemented. Possible choices are {REGULARIZERS_BY_TYPE.keys()}"

    regularizer: Regularizer = REGULARIZERS_BY_TYPE[type](regularizer_parameters)

    return regularizer


def create_regularizer_parameters(regularizer_dictionary: Dict[AnyStr, Any]) -> RegularizerParameters:
    """Create regularizer parameters."""
    type = regularizer_dictionary.pop("type")

    assert type in REGULARIZER_PARAMETERS_BY_TYPE.keys(), (
        f"Regularizer Type {type} is not implemented. "
        f"Possible choices are {REGULARIZER_PARAMETERS_BY_TYPE.keys()}"
    )

    data_class = REGULARIZER_PARAMETERS_BY_TYPE[type]
    regularizer_parameters = data_class(**regularizer_dictionary)
    return regularizer_parameters

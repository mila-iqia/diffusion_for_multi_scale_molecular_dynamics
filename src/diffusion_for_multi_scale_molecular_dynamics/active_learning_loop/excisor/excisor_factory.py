from typing import Any, AnyStr, Dict

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import (
    BaseEnvironmentExcision, BaseEnvironmentExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.nearest_neighbors_excisor import (
    NearestNeighborsExcision, NearestNeighborsExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.no_op_excisor import (
    NoOpExcision, NoOpExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.spherical_excisor import (
    SphericalExcision, SphericalExcisionArguments)

EXCISOR_PARAMETERS_BY_NAME = dict(
    noop=NoOpExcisionArguments,
    nearest_neighbors=NearestNeighborsExcisionArguments,
    spherical_cutoff=SphericalExcisionArguments,
)
EXCISOR_BY_NAME = dict(
    noop=NoOpExcision,
    nearest_neighbors=NearestNeighborsExcision,
    spherical_cutoff=SphericalExcision,
)


def create_excisor_parameters(
    excisor_dictionary: Dict[AnyStr, Any],
) -> BaseEnvironmentExcisionArguments:
    """Create atomic excision method parameters.

    Args:
        excisor_dictionary : parsed configuration for the excision.

    Returns:
        excisor_parameters: a configuration object for an excision object.
    """
    algorithm = excisor_dictionary["algorithm"]

    assert (
        algorithm in EXCISOR_PARAMETERS_BY_NAME.keys()
    ), f"Excision method {algorithm} is not implemented. Possible choices are {EXCISOR_PARAMETERS_BY_NAME.keys()}"

    excisor_parameters = EXCISOR_PARAMETERS_BY_NAME[algorithm](**excisor_dictionary)
    return excisor_parameters


def create_excisor(
    excisor_parameters: BaseEnvironmentExcisionArguments,
) -> BaseEnvironmentExcision:
    """Create an excisor.

    This is a factory method responsible for instantiating the excisor.
    """
    algorithm = excisor_parameters.algorithm
    assert (
        algorithm in EXCISOR_BY_NAME.keys()
    ), f"Excision method {algorithm} is not implemented. Possible choices are {EXCISOR_BY_NAME.keys()}"

    excisor = EXCISOR_BY_NAME[algorithm](excisor_parameters)

    return excisor

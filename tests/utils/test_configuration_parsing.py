from dataclasses import asdict, dataclass

import pytest

from diffusion_for_multi_scale_molecular_dynamics.utils.configuration_parsing import \
    create_parameters_from_configuration_dictionary


@dataclass(kw_only=True)
class DummyParameters:
    """Base dataclass for some set of parameters"""

    name: str


@dataclass(kw_only=True)
class FirstKindDummyParameters(DummyParameters):
    """Base dataclass for some set of parameters"""

    name: str = "first_kind"
    a: float = 1.0
    b: float = 2.0


@dataclass(kw_only=True)
class SecondKindDummyParameters(DummyParameters):
    """Base dataclass for some set of parameters"""

    name: str = "second_kind"
    x: float = 0.1
    y: float = 0.2
    z: float = 0.3


@pytest.fixture(params=[FirstKindDummyParameters, SecondKindDummyParameters])
def parameters(request):
    return request.param()


@pytest.fixture()
def configuration(parameters):
    return asdict(parameters)


@pytest.fixture()
def options():
    return dict(
        first_kind=FirstKindDummyParameters, second_kind=SecondKindDummyParameters
    )


def test_create_parameters_from_configuration_dictionary(
    configuration, options, parameters
):

    computed_parameters = create_parameters_from_configuration_dictionary(
        configuration=configuration, identifier="name", options=options
    )
    assert computed_parameters == parameters

from typing import Any, AnyStr, Dict

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.harmonic_force import (
    HarmonicForce, HarmonicForceParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.repulsive_force import (
    RepulsiveForce, RepulsiveForceParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.zbl_force import (
    ZBLForce, ZBLForceParameters)

REPULSIVE_FORCE_BY_ARCH = dict(
    harmonic=HarmonicForce,
    zbl=ZBLForce,
)

REPULSIVE_FORCE_PARAMETERS_BY_ARCH = dict(
    harmonic=HarmonicForceParameters,
    zbl=ZBLForceParameters,
)


def create_repulsive_force(repulsive_force_parameters: RepulsiveForceParameters):
    """Create Repulsive Force.

    This is a factory method responsible for instantiating the repulsive force.
    """
    architecture = repulsive_force_parameters.architecture
    assert (
        architecture in REPULSIVE_FORCE_BY_ARCH.keys()
    ), f"Architecture {architecture} is not implemented. Possible choices are {REPULSIVE_FORCE_BY_ARCH.keys()}"

    instantiated_repulsive_force: RepulsiveForce = REPULSIVE_FORCE_BY_ARCH[architecture](
        repulsive_force_parameters
    )
    return instantiated_repulsive_force


def create_repulsive_force_parameters(
    repulsive_force_dictionary: Dict[AnyStr, Any],
) -> RepulsiveForceParameters:
    """Create RepulsiveForceParameters from YAML dict."""
    assert (
        "architecture" in repulsive_force_dictionary
    ), "The architecture of the repulsive force must be specified."
    repulsive_force_architecture = repulsive_force_dictionary["architecture"]

    assert repulsive_force_architecture in REPULSIVE_FORCE_PARAMETERS_BY_ARCH.keys(), (
        f"Architecture {repulsive_force_architecture} is not implemented. "
        f"Possible choices are {REPULSIVE_FORCE_PARAMETERS_BY_ARCH.keys()}"
    )

    params_cls = REPULSIVE_FORCE_PARAMETERS_BY_ARCH[repulsive_force_architecture]
    return params_cls(**repulsive_force_dictionary)

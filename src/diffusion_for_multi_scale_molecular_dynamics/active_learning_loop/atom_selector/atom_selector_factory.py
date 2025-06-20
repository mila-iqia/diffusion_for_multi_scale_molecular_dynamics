from typing import Any, AnyStr, Dict

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.base_atom_selector import (
    BaseAtomSelector, BaseAtomSelectorParameters)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.threshold_atom_selector import (
    ThresholdAtomSelector, ThresholdAtomSelectorParameters)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.top_k_atom_selector import (
    TopKAtomSelector, TopKAtomSelectorParameters)

ATOM_SELECTOR_PARAMETERS_BY_NAME = dict(
    threshold=ThresholdAtomSelectorParameters,
    top_k=TopKAtomSelectorParameters,
)

ATOM_SELECTOR_BY_NAME = dict(
    threshold=ThresholdAtomSelector,
    top_k=TopKAtomSelector,
)


def create_atom_selector_parameters(atom_selector_parameter_dictionary: Dict[AnyStr, Any]) \
        -> BaseAtomSelectorParameters:
    """Create atom selector parameters.

    This factory method is responsible for creating an atom selector configuration based on configuration parameters.

    Args:
        atom_selector_parameter_dictionary: a dictionary of relevant parameters to instantiate an atom selector.

    Returns:
        atom_selector_parameters: a configured atom selector parameter object.
    """
    assert "algorithm" in atom_selector_parameter_dictionary, "The algorithm is missing."
    algorithm = atom_selector_parameter_dictionary["algorithm"]

    assert (
        algorithm in ATOM_SELECTOR_PARAMETERS_BY_NAME.keys()
    ), (f"Atom selector method {algorithm} is not implemented. "
        f"Possible choices are {ATOM_SELECTOR_PARAMETERS_BY_NAME.keys()}")

    atom_selector_parameters = ATOM_SELECTOR_PARAMETERS_BY_NAME[algorithm](**atom_selector_parameter_dictionary)
    return atom_selector_parameters


def create_atom_selector(atom_selector_parameters: BaseAtomSelectorParameters) -> BaseAtomSelector:
    """Create atom selector.

    This factor method is responsible for creating an atom selector based on configuration parameters.

    Args:
        atom_selector_parameters: a configured atom selector parameter object.

    Returns:
        atom_selector: an instance of an atom selector.
    """
    atom_selector = ATOM_SELECTOR_BY_NAME[atom_selector_parameters.algorithm](atom_selector_parameters)
    return atom_selector

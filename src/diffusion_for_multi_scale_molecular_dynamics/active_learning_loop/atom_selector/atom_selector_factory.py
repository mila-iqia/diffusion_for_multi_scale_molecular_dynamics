from typing import Any, AnyStr, Dict

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.base_atom_selector import \
    BaseAtomSelector
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


def create_atom_selector(atom_selector_parameter_dictionary: Dict[AnyStr, Any]) -> BaseAtomSelector:
    """Create atom selector.

    This factor method is responsible for creating an atom selector based on configuration parameters.

    Args:
        atom_selector_parameter_dictionary: a dictionary of relevant parameters to instantiate an atom selector.

    Returns:
        atom_selector: a instance of an atom selector.
    """
    assert "algorithm" in atom_selector_parameter_dictionary, "The algorithm is missing."
    algorithm = atom_selector_parameter_dictionary["algorithm"]

    assert (
        algorithm in ATOM_SELECTOR_PARAMETERS_BY_NAME.keys()
    ), (f"Atom selector method {algorithm} is not implemented. "
        f"Possible choices are {ATOM_SELECTOR_PARAMETERS_BY_NAME.keys()}")

    atom_selector_parameters = ATOM_SELECTOR_PARAMETERS_BY_NAME[algorithm](**atom_selector_parameter_dictionary)
    atom_selector = ATOM_SELECTOR_BY_NAME[algorithm](atom_selector_parameters)

    return atom_selector

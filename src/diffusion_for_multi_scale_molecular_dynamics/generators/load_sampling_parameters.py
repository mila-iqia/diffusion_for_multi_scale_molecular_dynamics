from typing import Any, AnyStr, Dict

from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import \
    SamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.ode_position_generator import \
    ODESamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.sde_position_generator import \
    SDESamplingParameters


def load_sampling_parameters(
    sampling_parameter_dictionary: Dict[AnyStr, Any]
) -> SamplingParameters:
    """Load sampling parameters.

    Extract the needed information from the configuration dictionary.

    Args:
        sampling_parameter_dictionary: dictionary of hyperparameters loaded from a config file

    Returns:
        sampling_parameters: the relevant configuration object.
    """
    assert (
        "algorithm" in sampling_parameter_dictionary
    ), "The sampling parameters must select an algorithm."
    algorithm = sampling_parameter_dictionary["algorithm"]

    assert algorithm in [
        "ode",
        "sde",
        "predictor_corrector",
    ], "Unknown algorithm. Possible choices are 'ode', 'sde' and 'predictor_corrector'"

    match algorithm:
        case "predictor_corrector":
            sampling_parameters = PredictorCorrectorSamplingParameters(
                **sampling_parameter_dictionary
            )
        case "ode":
            sampling_parameters = ODESamplingParameters(**sampling_parameter_dictionary)
        case "sde":
            sampling_parameters = SDESamplingParameters(**sampling_parameter_dictionary)
        case _:
            raise NotImplementedError(f"algorithm '{algorithm}' is not implemented")

    return sampling_parameters

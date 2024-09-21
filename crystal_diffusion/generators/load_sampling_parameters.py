from typing import Any, AnyStr, Dict

from crystal_diffusion.generators.ode_position_generator import \
    ODESamplingParameters
from crystal_diffusion.generators.position_generator import SamplingParameters
from crystal_diffusion.generators.predictor_corrector_position_generator import \
    PredictorCorrectorSamplingParameters
from crystal_diffusion.generators.sde_position_generator import \
    SDESamplingParameters


def load_sampling_parameters(sampling_parameter_dictionary: Dict[AnyStr, Any]) -> SamplingParameters:
    """Load sampling parameters.

    Extract the needed information from the configuration dictionary.

    Args:
        sampling_parameter_dictionary: dictionary of hyperparameters loaded from a config file

    Returns:
        sampling_parameters: the relevant configuration object.
    """
    assert 'algorithm' in sampling_parameter_dictionary, "The sampling parameters must select an algorithm."
    algorithm = sampling_parameter_dictionary['algorithm']

    assert algorithm in ['ode', 'sde', 'predictor_corrector'], \
        "Unknown algorithm. Possible choices are 'ode', 'sde' and 'predictor_corrector'"

    match algorithm:
        case 'predictor_corrector':
            sampling_parameters = PredictorCorrectorSamplingParameters(**sampling_parameter_dictionary)
        case 'ode':
            sampling_parameters = ODESamplingParameters(**sampling_parameter_dictionary)
        case 'sde':
            sampling_parameters = SDESamplingParameters(**sampling_parameter_dictionary)
        case _:
            raise NotImplementedError(f"algorithm '{algorithm}' is not implemented")

    return sampling_parameters

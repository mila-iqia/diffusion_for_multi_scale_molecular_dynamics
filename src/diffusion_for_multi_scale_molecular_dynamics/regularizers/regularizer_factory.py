from typing import Any, AnyStr, Dict

from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import \
    AnalyticalScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.regularizers.analytical_regression_regularizer import (
    AnalyticalRegressionRegularizer, AnalyticalRegressionRegularizerParameters)
from diffusion_for_multi_scale_molecular_dynamics.regularizers.consistency_regularizer import (
    ConsistencyRegularizer, ConsistencyRegularizerParameters)
from diffusion_for_multi_scale_molecular_dynamics.regularizers.fokker_planck_regularizer import (
    FokkerPlanckRegularizer, FokkerPlanckRegularizerParameters)
from diffusion_for_multi_scale_molecular_dynamics.regularizers.regularizer import (
    Regularizer, RegularizerParameters)

REGULARIZERS_BY_TYPE = dict(
    fokker_planck=FokkerPlanckRegularizer,
    analytical_regression=AnalyticalRegressionRegularizer,
    consistency=ConsistencyRegularizer,
)

REGULARIZER_PARAMETERS_BY_TYPE = dict(
    fokker_planck=FokkerPlanckRegularizerParameters,
    analytical_regression=AnalyticalRegressionRegularizerParameters,
    consistency=ConsistencyRegularizerParameters,
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

    match type:
        case "analytical_regression":
            analytical_score_network_parameters = (
                AnalyticalScoreNetworkParameters(**regularizer_dictionary.pop('analytical_score_network')))
            regularizer_parameters = data_class(**regularizer_dictionary,
                                                analytical_score_network_parameters=analytical_score_network_parameters)
        case "consistency":
            noise_parameters = NoiseParameters(**regularizer_dictionary.pop('noise'))
            sampling_parameters = PredictorCorrectorSamplingParameters(**regularizer_dictionary.pop('sampling'))

            if 'analytical_score_network' in regularizer_dictionary:
                analytical_score_network_parameters = (
                    AnalyticalScoreNetworkParameters(**regularizer_dictionary.pop('analytical_score_network')))
            else:
                analytical_score_network_parameters = None

            regularizer_parameters = data_class(**regularizer_dictionary,
                                                noise_parameters=noise_parameters,
                                                sampling_parameters=sampling_parameters,
                                                analytical_score_network_parameters=analytical_score_network_parameters)
        case _:
            regularizer_parameters = data_class(**regularizer_dictionary)

    return regularizer_parameters

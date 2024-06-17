"""Functions to instantiate a model based on the provided hyperparameters."""
import dataclasses
import logging
from typing import Any, AnyStr, Dict

from crystal_diffusion.models.loss import LossParameters
from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                ValidOptimizerName)
from crystal_diffusion.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from crystal_diffusion.models.scheduler import get_scheduler_parameters
from crystal_diffusion.models.score_networks.diffusion_mace_score_network import \
    DiffusionMACEScoreNetworkParameters
from crystal_diffusion.models.score_networks.mace_score_network import \
    MACEScoreNetworkParameters
from crystal_diffusion.models.score_networks.mlp_score_network import (
    MLPScoreNetwork, MLPScoreNetworkParameters)
from crystal_diffusion.models.score_networks.score_network import \
    ScoreNetworkParameters
from crystal_diffusion.models.score_networks.score_prediction_head import (
    MaceEquivariantScorePredictionHeadParameters,
    MaceMLPScorePredictionHeadParameters)
from crystal_diffusion.samplers.variance_sampler import NoiseParameters

logger = logging.getLogger(__name__)


def load_diffusion_model(hyper_params: Dict[AnyStr, Any]) -> PositionDiffusionLightningModel:
    """Load a position diffusion model from the hyperparameters.

    Args:
        hyper_params: dictionary of hyperparameters loaded from a config file

    Returns:
        Diffusion model randomly initialized
    """
    global_parameters = dict(max_atom=hyper_params['data']['max_atom'],
                             spatial_dimension=hyper_params.get('spatial_dimension', 3))

    score_network_dictionary = hyper_params['model']['score_network']
    score_network_parameters = extract_score_network_parameters(score_network_dictionary, global_parameters)

    hyper_params['optimizer']['name'] = ValidOptimizerName(hyper_params['optimizer']['name'])

    optimizer_parameters = OptimizerParameters(
        **hyper_params['optimizer']
    )

    scheduler_parameters = get_scheduler_parameters(hyper_params)

    if "loss" in hyper_params['model']:
        loss_parameters = LossParameters(**hyper_params['model']['loss'])
    else:
        loss_parameters = LossParameters()

    noise_parameters = NoiseParameters(**hyper_params['model']['noise'])

    diffusion_params = PositionDiffusionParameters(
        score_network_parameters=score_network_parameters,
        loss_parameters=loss_parameters,
        optimizer_parameters=optimizer_parameters,
        scheduler_parameters=scheduler_parameters,
        noise_parameters=noise_parameters,
    )

    model = PositionDiffusionLightningModel(diffusion_params)
    logger.info('model info:\n' + str(model) + '\n')

    return model


def extract_score_network_parameters(score_network_dictionary: Dict[AnyStr, Any],
                                     global_parameters: Dict[AnyStr, Any]) -> ScoreNetworkParameters:
    """Extract the score network parameters.

    Args:
        score_network_dictionary : input parameters that describe the score network.
        global_parameters : global hyperparameters.

    Returns:
        score_network_parameters: the dataclass configuration object describing the score network.
    """
    assert 'architecture' in score_network_dictionary, "The architecture of the score network must be specified."
    score_network_architecture = score_network_dictionary['architecture']

    # TODO: this is a hot mess. Surely there is a better way.
    if score_network_architecture == 'mlp':
        score_network_parameters_class = MLPScoreNetworkParameters
        left_over_parameters = dict(score_network_dictionary)
    elif score_network_architecture == 'diffusion_mace':
        score_network_parameters_class = DiffusionMACEScoreNetworkParameters
        left_over_parameters = dict(score_network_dictionary)
    elif score_network_architecture == 'mace':
        score_network_parameters_class = MACEScoreNetworkParameters
        assert 'prediction_head_parameters' in score_network_dictionary, \
            "The prediction head parameters must be specified."

        prediction_head_parameters_dict = score_network_dictionary['prediction_head_parameters']
        head_name = prediction_head_parameters_dict['name']

        if head_name == 'mlp':
            prediction_head_parameters = MaceMLPScorePredictionHeadParameters(**prediction_head_parameters_dict)
        elif head_name == 'equivariant':
            prediction_head_parameters = MaceEquivariantScorePredictionHeadParameters(**prediction_head_parameters_dict)
        else:
            raise NotImplementedError(f'Prediction head {head_name} is not available.')

        left_over_parameters = dict(score_network_dictionary)
        # Overload the head parameters by the corresponding dataclass object.
        left_over_parameters['prediction_head_parameters'] = prediction_head_parameters

    else:
        raise NotImplementedError(f'Architecture {score_network_architecture} is not available.')

    input = dict(left_over_parameters)
    for name, value in global_parameters.items():
        if name not in [field.name for field in dataclasses.fields(score_network_parameters_class)]:
            continue

        if name in input:
            assert input[name] == value, f"inconsistent configuration values for {name}"
        else:
            input[name] = value

    score_network_parameters = score_network_parameters_class(**input)

    return score_network_parameters


def load_model(hyper_params):  # pragma: no cover
    """Instantiate a model.

    Args:
        hyper_params (dict): hyper parameters from the config file

    Returns:
        model (obj): A neural network model object.
    """
    architecture = hyper_params['architecture']
    # __TODO__ fix architecture list
    if architecture == 'simple_mlp':
        model_class = MLPScoreNetwork
    else:
        raise ValueError('architecture {} not supported'.format(architecture))
    logger.info('selected architecture: {}'.format(architecture))

    model = model_class(hyper_params)
    logger.info('model info:\n' + str(model) + '\n')

    return model

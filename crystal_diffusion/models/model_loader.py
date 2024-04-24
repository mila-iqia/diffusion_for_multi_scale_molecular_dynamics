"""Functions to instantiate a model based on the provided hyperparameters."""
import logging
from typing import Any, AnyStr, Dict

from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                ValidOptimizerNames)
from crystal_diffusion.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from crystal_diffusion.models.score_network import (MACEScoreNetworkParameters,
                                                    MLPScoreNetwork,
                                                    MLPScoreNetworkParameters)
from crystal_diffusion.samplers.variance_sampler import NoiseParameters

logger = logging.getLogger(__name__)


def load_diffusion_model(hyper_params: Dict[AnyStr, Any]) -> PositionDiffusionLightningModel:
    """Load a position diffusion model from the hyperparameters.

    Args:
        hyper_params: dictionary of hyperparameters loaded from a config file

    Returns:
        Diffusion model randomly initialized
    """
    score_network_architecture = hyper_params['model']['score_network'].get('architecture', 'mlp')
    if score_network_architecture == 'mlp':
        score_network_parameters_class = MLPScoreNetworkParameters
    elif score_network_architecture == 'mace':
        score_network_parameters_class = MACEScoreNetworkParameters
    else:
        raise NotImplementedError(f'Architecture {score_network_architecture} is not available.')

    score_network_parameters = score_network_parameters_class(
        number_of_atoms=hyper_params['data']['max_atom'],
        **{k: v for k, v in hyper_params['model']['score_network'].items() if k != 'architecture'}
    )

    score_network_parameters.spatial_dimension = hyper_params.get('spatial_dimension', 3)

    hyper_params['optimizer']['name'] = ValidOptimizerNames(hyper_params['optimizer']['name'])

    optimizer_parameters = OptimizerParameters(
        **hyper_params['optimizer']
    )

    noise_parameters = NoiseParameters(**hyper_params['model']['noise'])

    diffusion_params = PositionDiffusionParameters(
        score_network_parameters=score_network_parameters,
        optimizer_parameters=optimizer_parameters,
        noise_parameters=noise_parameters,
    )

    model = PositionDiffusionLightningModel(diffusion_params, score_network_architecture=score_network_architecture)
    logger.info('model info:\n' + str(model) + '\n')

    return model


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

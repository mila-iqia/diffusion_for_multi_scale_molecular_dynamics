"""Functions to instantiate a model based on the provided hyperparameters."""
import logging
from typing import Any, AnyStr, Dict

from crystal_diffusion.models.loss import LossParameters
from crystal_diffusion.models.optimizer import create_optimizer_parameters
from crystal_diffusion.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from crystal_diffusion.models.scheduler import get_scheduler_parameters
from crystal_diffusion.models.score_networks import \
    create_score_network_parameters
from crystal_diffusion.samplers.variance_sampler import NoiseParameters

logger = logging.getLogger(__name__)


def load_diffusion_model(hyper_params: Dict[AnyStr, Any]) -> PositionDiffusionLightningModel:
    """Load a position diffusion model from the hyperparameters.

    Args:
        hyper_params: dictionary of hyperparameters loaded from a config file

    Returns:
        Diffusion model randomly initialized
    """
    globals_dict = dict(max_atom=hyper_params['data']['max_atom'],
                        spatial_dimension=hyper_params.get('spatial_dimension', 3))

    score_network_dictionary = hyper_params['model']['score_network']
    score_network_parameters = create_score_network_parameters(score_network_dictionary, globals_dict)

    optimizer_configuration_dictionary = hyper_params['optimizer']
    optimizer_parameters = create_optimizer_parameters(optimizer_configuration_dictionary)

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

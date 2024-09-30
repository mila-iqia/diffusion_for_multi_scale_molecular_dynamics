"""Functions to instantiate a model based on the provided hyperparameters."""
import logging
from typing import Any, AnyStr, Dict

from crystal_diffusion.models.loss import create_loss_parameters
from crystal_diffusion.models.optimizer import create_optimizer_parameters
from crystal_diffusion.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from crystal_diffusion.models.scheduler import create_scheduler_parameters
from crystal_diffusion.models.score_networks.score_network_factory import \
    create_score_network_parameters
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.samples.diffusion_sampling_parameters import \
    load_diffusion_sampling_parameters

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

    score_network_dict = hyper_params['model']['score_network']
    score_network_parameters = create_score_network_parameters(score_network_dict, globals_dict)

    optimizer_configuration_dict = hyper_params['optimizer']
    optimizer_parameters = create_optimizer_parameters(optimizer_configuration_dict)

    scheduler_parameters = create_scheduler_parameters(hyper_params)

    model_dict = hyper_params['model']
    loss_parameters = create_loss_parameters(model_dict)

    noise_dict = model_dict['noise']
    noise_parameters = NoiseParameters(**noise_dict)

    diffusion_sampling_parameters = load_diffusion_sampling_parameters(hyper_params)

    diffusion_params = PositionDiffusionParameters(
        score_network_parameters=score_network_parameters,
        loss_parameters=loss_parameters,
        optimizer_parameters=optimizer_parameters,
        scheduler_parameters=scheduler_parameters,
        noise_parameters=noise_parameters,
        diffusion_sampling_parameters=diffusion_sampling_parameters
    )

    model = PositionDiffusionLightningModel(diffusion_params)
    logger.info('model info:\n' + str(model) + '\n')

    return model

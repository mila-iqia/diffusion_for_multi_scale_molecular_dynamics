"""Functions to instantiate a model based on the provided hyperparameters."""

import logging
from typing import Any, AnyStr, Dict

import torch

from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import \
    create_loss_parameters
from diffusion_for_multi_scale_molecular_dynamics.models.axl_diffusion_lightning_model import (
    AXLDiffusionLightningModel, AXLDiffusionParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.optimizer import \
    create_optimizer_parameters
from diffusion_for_multi_scale_molecular_dynamics.models.scheduler import \
    create_scheduler_parameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network_factory import \
    create_score_network_parameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noisers.lattice_noiser import \
    LatticeDataParameters
from diffusion_for_multi_scale_molecular_dynamics.oracle.energy_oracle_factory import \
    create_energy_oracle_parameters
from diffusion_for_multi_scale_molecular_dynamics.regularizers.regularizer_factory import \
    create_regularizer_parameters
from diffusion_for_multi_scale_molecular_dynamics.sampling.diffusion_sampling_parameters import \
    load_diffusion_sampling_parameters

logger = logging.getLogger(__name__)


def load_diffusion_model(hyper_params: Dict[AnyStr, Any]) -> AXLDiffusionLightningModel:
    """Load a position diffusion model from the hyperparameters.

    Args:
        hyper_params: dictionary of hyperparameters loaded from a config file

    Returns:
        Diffusion model randomly initialized
    """
    elements = hyper_params["elements"]
    globals_dict = dict(
        max_atom=hyper_params["data"]["max_atom"],
        spatial_dimension=hyper_params.get("spatial_dimension", 3),
        elements=elements,
    )

    score_network_dict = hyper_params["model"]["score_network"]
    score_network_parameters = create_score_network_parameters(
        score_network_dict, globals_dict
    )

    optimizer_configuration_dict = hyper_params["optimizer"]
    optimizer_parameters = create_optimizer_parameters(optimizer_configuration_dict)

    scheduler_parameters = create_scheduler_parameters(hyper_params)

    model_dict = hyper_params["model"]
    loss_parameters = create_loss_parameters(model_dict)

    noise_dict = model_dict["noise"]
    noise_parameters = NoiseParameters(**noise_dict)

    diffusion_sampling_parameters = load_diffusion_sampling_parameters(hyper_params)

    oracle_parameters = None
    if "oracle" in hyper_params:
        oracle_parameters = create_energy_oracle_parameters(
            hyper_params["oracle"], elements
        )

    regularizer_parameters = None
    if "regularizer" in hyper_params:
        regularizer_parameters = create_regularizer_parameters(hyper_params["regularizer"])

    lattice_parameters = dict(
        spatial_dimension=globals_dict["spatial_dimension"],
    )
    lattice_parameters = LatticeDataParameters(**lattice_parameters)

    diffusion_params = AXLDiffusionParameters(
        score_network_parameters=score_network_parameters,
        loss_parameters=loss_parameters,
        optimizer_parameters=optimizer_parameters,
        scheduler_parameters=scheduler_parameters,
        noise_parameters=noise_parameters,
        regularizer_parameters=regularizer_parameters,
        diffusion_sampling_parameters=diffusion_sampling_parameters,
        oracle_parameters=oracle_parameters,
        lattice_parameters=lattice_parameters,
    )

    model = AXLDiffusionLightningModel(diffusion_params)
    logger.info("model info:\n" + str(model) + "\n")

    return model

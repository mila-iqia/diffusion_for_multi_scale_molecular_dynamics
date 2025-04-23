"""Functions to instantiate a data loader based on the provided hyperparameters."""
import argparse
import logging
from typing import Any, AnyStr, Dict

import lightning as pl

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.gaussian_data_module import (
    GaussianDataModule, GaussianDataModuleParameters)
from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import (
    LammpsDataModuleParameters, LammpsForDiffusionDataModule)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters

logger = logging.getLogger(__name__)


def load_data_module(hyper_params: Dict[AnyStr, Any], args: argparse.Namespace) -> pl.LightningDataModule:
    """Load data module.

    This method creates the data module based on configuration and input arguments.

    Args:
        hyper_params: configuration parameters.
        args: parsed command line arguments.

    Returns:
        data_module:  the data module corresponding to the configuration and input arguments.
    """
    assert 'data' in hyper_params, \
        "The configuration should contain a 'data' block describing the data source."

    data_config = hyper_params["data"]
    data_source = data_config.pop("data_source", "LAMMPS")
    noise = data_config.pop("noise")
    noise_parameters = NoiseParameters(**noise)

    match data_source:
        case "LAMMPS":
            data_params = LammpsDataModuleParameters(**data_config,
                                                     noise_parameters=noise_parameters,
                                                     elements=hyper_params["elements"])
            data_module = LammpsForDiffusionDataModule(hyper_params=data_params,
                                                       lammps_run_dir=args.data,
                                                       processed_dataset_dir=args.processed_datadir,
                                                       working_cache_dir=args.dataset_working_dir)

        case "gaussian":
            data_params = GaussianDataModuleParameters(**data_config,
                                                       noise_parameters=noise_parameters,
                                                       elements=hyper_params["elements"])
            data_module = GaussianDataModule(data_params)
        case _:
            raise NotImplementedError(
                f"Data source '{data_source}' is not implemented"
            )

    return data_module

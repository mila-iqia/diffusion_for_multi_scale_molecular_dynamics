"""Overfit fake data.

A simple sanity check experiment to check the learning behavior of the position diffusion model.
The training data is taken to be a large batch of identical configurations composed of one atom in 1D at x0.
This highly artificial case is useful to sanity check that the code behaves as expected:
 -  the loss should converge towards zero
 -  the trained score network should reproduce the perturbation kernel, at least in the regions where it is sampled.
 -  the generated samples should be tightly clustered around x0.
"""
import dataclasses
import os

import pytorch_lightning
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from crystal_diffusion.models.optimizer import OptimizerParameters
from crystal_diffusion.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from crystal_diffusion.models.score_networks.mlp_score_network import \
    MLPScoreNetworkParameters
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from sanity_checks import SANITY_CHECK_FOLDER
from sanity_checks.sanity_check_callbacks import (
    TensorboardGeneratedSamplesLoggingCallback,
    TensorboardHistogramLoggingCallback, TensorboardSamplesLoggingCallback,
    TensorboardScoreAndErrorLoggingCallback)


class HPLoggingCallback(Callback):
    """This callback is responsible for logging hyperparameters."""

    def on_train_start(self, trainer, pl_module):
        """Log hyperparameters when training starts."""
        assert hasattr(
            pl_module, "hyper_params"
        ), "The lightning module should have a hyper_params attribute for HP logging."
        hp_dict = dataclasses.asdict(pl_module.hyper_params)
        trainer.logger.log_hyperparams(hp_dict)


batch_size = 4096
number_of_atoms = 1
spatial_dimension = 1
total_time_steps = 100
number_of_corrector_steps = 1

x0 = 0.5

sigma_min = 0.005
sigma_max = 0.5

lr = 0.001
max_epochs = 2000

hidden_dimensions = [64, 128, 256]


score_network_parameters = MLPScoreNetworkParameters(
    number_of_atoms=number_of_atoms,
    hidden_dimensions=hidden_dimensions,
    spatial_dimension=spatial_dimension,
)

optimizer_parameters = OptimizerParameters(name="adam", learning_rate=lr)

noise_parameters = NoiseParameters(total_time_steps=total_time_steps, sigma_min=sigma_min, sigma_max=sigma_max)

hyper_params = PositionDiffusionParameters(
    score_network_parameters=score_network_parameters,
    optimizer_parameters=optimizer_parameters,
    noise_parameters=noise_parameters,
)

generated_samples_callback = (
    TensorboardGeneratedSamplesLoggingCallback(noise_parameters=noise_parameters,
                                               number_of_corrector_steps=number_of_corrector_steps,
                                               score_network_parameters=score_network_parameters,
                                               number_of_samples=1024))

score_error_callback = TensorboardScoreAndErrorLoggingCallback(x0=x0)

tbx_logger = TensorBoardLogger(save_dir=os.path.join(SANITY_CHECK_FOLDER, "tensorboard"), name="overfit_fake_data")

if __name__ == '__main__':
    pytorch_lightning.seed_everything(123)
    all_positions = x0 * torch.ones(batch_size, number_of_atoms, spatial_dimension)
    data = [dict(relative_positions=configuration) for configuration in all_positions]
    train_dataloader = DataLoader(data, batch_size=batch_size)

    lightning_model = PositionDiffusionLightningModel(hyper_params)

    trainer = Trainer(accelerator='cpu',
                      max_epochs=max_epochs,
                      logger=tbx_logger,
                      log_every_n_steps=25,
                      callbacks=[HPLoggingCallback(),
                                 generated_samples_callback,
                                 score_error_callback,
                                 TensorboardHistogramLoggingCallback(),
                                 TensorboardSamplesLoggingCallback(),
                                 LearningRateMonitor(logging_interval='step')])
    trainer.fit(lightning_model, train_dataloaders=train_dataloader)

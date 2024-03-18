"""Overfit fake data.

A simple sanity check experiment to verify that we can overfit a batch of random data.
"""
import os

import pytorch_lightning
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                ValidOptimizerNames)
from crystal_diffusion.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from crystal_diffusion.models.score_network import MLPScoreNetworkParameters
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from sanity_checks import SANITY_CHECK_FOLDER

batch_size = 16
number_of_atoms = 4
spatial_dimension = 2

score_network_parameters = MLPScoreNetworkParameters(
    number_of_atoms=number_of_atoms,
    hidden_dim=32,
    spatial_dimension=spatial_dimension,
)

optimizer_parameters = OptimizerParameters(name=ValidOptimizerNames("adam"),
                                           learning_rate=0.01)

noise_parameters = NoiseParameters(total_time_steps=10)

hyper_params = PositionDiffusionParameters(
    score_network_parameters=score_network_parameters,
    optimizer_parameters=optimizer_parameters,
    noise_parameters=noise_parameters,
)


tbx_logger = TensorBoardLogger(save_dir=os.path.join(SANITY_CHECK_FOLDER, "tensorboard"), name="overfit_fake_data")

if __name__ == '__main__':

    pytorch_lightning.seed_everything(123)

    all_positions = torch.rand(batch_size, number_of_atoms, spatial_dimension)
    data = [dict(relative_positions=configuration) for configuration in all_positions]
    train_dataloader = DataLoader(data, batch_size=batch_size)

    lightning_model = PositionDiffusionLightningModel(hyper_params)

    trainer = Trainer(accelerator='cpu', max_epochs=10000, logger=tbx_logger, log_every_n_steps=1)
    trainer.fit(lightning_model, train_dataloaders=train_dataloader)

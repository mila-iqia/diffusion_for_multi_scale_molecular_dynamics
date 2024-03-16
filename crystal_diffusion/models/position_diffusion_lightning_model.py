import logging
import typing
from dataclasses import dataclass

import pytorch_lightning as pl
import torch

from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                load_optimizer)
from crystal_diffusion.models.score_network import (MLPScoreNetwork,
                                                    MLPScoreNetworkParameters)
from crystal_diffusion.samplers.noisy_position_sampler import \
    NoisyPositionSampler
from crystal_diffusion.samplers.variance_sampler import (
    ExplodingVarianceSampler, NoiseParameters)
from crystal_diffusion.score.wrapped_gaussian_score import \
    get_sigma_normalized_score
from crystal_diffusion.utils.tensor_utils import \
    broadcast_batch_tensor_to_all_dimensions

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class PositionDiffusionParameters:
    """Position Diffusion parameters."""
    score_network_parameters: MLPScoreNetworkParameters
    optimizer_parameters: OptimizerParameters
    noise_parameters: NoiseParameters
    kmax_target_score: int


class PositionDiffusionLightningModel(pl.LightningModule):
    """Position Diffusion Lightning Model.

    This lightning model can train a score network predict the noise for relative positions.

    TODO : filling this class with what is needed to train a diffusion model.
    """

    def __init__(self, hyper_params: PositionDiffusionParameters):
        """Init method.

        This initializes the class.
        """
        super().__init__()

        self.hyper_params = hyper_params

        # we will model sigma x score
        self.sigma_normalized_score_network = MLPScoreNetwork(hyper_params.score_network_parameters)

        self.noisy_position_sampler = NoisyPositionSampler()
        self.variance_sampler = ExplodingVarianceSampler(hyper_params.noise_parameters)

    def configure_optimizers(self):
        """Returns the combination of optimizer(s) and learning rate scheduler(s) to train with.

        Here, we read all the optimization-related hyperparameters from the config dictionary and
        create the required optimizer/scheduler combo.

        This function will be called automatically by the pytorch lightning trainer implementation.
        See https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html for more info
        on the expected returned elements.
        """
        return load_optimizer(self.hyper_params.optimizer_parameters, self)

    def _generic_step(
        self,
        batch: typing.Any,
        batch_idx: int,
    ) -> typing.Any:
        """Runs the prediction + evaluation step for training/validation/testing."""
        # The relative positions have dimensions [batch_size, number of atoms, spatial dimension].
        real_relative_positions = batch["relative_positions"]
        shape = real_relative_positions.shape
        batch_size = shape[0]

        noise_sample = self.variance_sampler.get_random_noise_sample(batch_size)
        # batch_sigma_values has dimension [batch_size]
        batch_sigma_values = noise_sample.sigma

        # broadcast the sigma values to be of shape [batch_size, number of atoms, spatial dimension].
        # This can be interpreted as [batch_size, (configuration)]. The sigma value must be the
        # same for a given "configuration".
        sigmas = broadcast_batch_tensor_to_all_dimensions(batch_values=batch_sigma_values, final_shape=shape)

        noisy_relative_positions = self.noisy_position_sampler.get_noisy_position_sample(real_relative_positions,
                                                                                         sigmas)

        delta_positions = torch.remainder(noisy_relative_positions - real_relative_positions, 1.0)
        target_normalized_scores = get_sigma_normalized_score(delta_positions,
                                                              sigmas,
                                                              kmax=self.hyper_params.kmax_target_score)

        pos_key = self.sigma_normalized_score_network.position_key
        time_key = self.sigma_normalized_score_network.timestep_key
        augmented_batch = {pos_key: noisy_relative_positions, time_key: noise_sample.time.reshape(-1, 1)}
        predicted_normalized_scores = self.sigma_normalized_score_network(augmented_batch)

        loss = torch.nn.functional.mse_loss(predicted_normalized_scores, target_normalized_scores, reduction='mean')
        return loss

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("epoch", self.current_epoch)
        self.log("step", self.global_step)
        return loss  # this function is required, as the loss returned here is used for backprop

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """Runs a prediction step for testing, logging the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("test_loss", loss)

import logging
import typing
from dataclasses import dataclass

import pytorch_lightning as pl
import torch

from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                load_optimizer)
from crystal_diffusion.models.scheduler import (SchedulerParameters,
                                                load_scheduler_dictionary)
from crystal_diffusion.models.score_network import (MACEScoreNetwork,
                                                    MACEScoreNetworkParameters,
                                                    MLPScoreNetwork,
                                                    MLPScoreNetworkParameters)
from crystal_diffusion.samplers.noisy_position_sampler import (
    NoisyPositionSampler, map_positions_to_unit_cell)
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

    score_network_parameters: typing.Union[MLPScoreNetworkParameters, MACEScoreNetworkParameters]
    optimizer_parameters: OptimizerParameters
    scheduler_parameters: typing.Union[SchedulerParameters, None] = None
    noise_parameters: NoiseParameters
    kmax_target_score: int = (
        4  # convergence parameter for the Ewald-like sum of the perturbation kernel.
    )


class PositionDiffusionLightningModel(pl.LightningModule):
    """Position Diffusion Lightning Model.

    This lightning model can train a score network predict the noise for relative positions.
    """

    def __init__(self, hyper_params: PositionDiffusionParameters, score_network_architecture: str = 'mlp'):
        """Init method.

        This initializes the class.
        """
        super().__init__()

        self.hyper_params = hyper_params
        self.save_hyperparameters(logger=False)  # It is not the responsibility of this class to log its parameters.

        # we will model sigma x score
        if score_network_architecture == 'mlp':
            score_network = MLPScoreNetwork
        elif score_network_architecture == 'mace':
            score_network = MACEScoreNetwork
        else:
            raise NotImplementedError(f'Architecture {score_network_architecture} is not implemented.')

        self.sigma_normalized_score_network = score_network(
            hyper_params.score_network_parameters
        )

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
        optimizer = load_optimizer(self.hyper_params.optimizer_parameters, self)
        output = dict(optimizer=optimizer)

        if self.hyper_params.scheduler_parameters is not None:
            scheduler_dict = load_scheduler_dictionary(hyper_params=self.hyper_params.scheduler_parameters,
                                                       optimizer=optimizer)
            output.update(scheduler_dict)

        return output

    @staticmethod
    def _get_batch_size(batch: torch.Tensor) -> int:
        """Get batch size.

        Args:
            batch : a dictionary that should contain a data sample.

        Returns:
            batch_size: the size of the batch.
        """
        # The relative positions have dimensions [batch_size, number_of_atoms, spatial_dimension].
        assert "relative_positions" in batch, "The field 'relative_positions' is missing from the input."
        batch_size = batch["relative_positions"].shape[0]
        return batch_size

    def _generic_step(
        self,
        batch: typing.Any,
        batch_idx: int,
    ) -> typing.Any:
        """Generic step.

        This "generic step" computes the loss for any of the possible lightning "steps".

        The  loss is defined as:
            L = 1 / T int_0^T dt lambda(t) E_{x0 ~ p_data} E_{xt~ p_{t| 0}}
                    [|S_theta(xt, t) - nabla_{xt} log p_{t | 0} (xt | x0)|^2]

            Where
                      T     : time range of the noising process
                  S_theta   : score network
                 p_{t| 0}   : perturbation kernel
                nabla log p : the target score
                lambda(t)   : is arbitrary, but chosen for convenience.

        In this implementation, we choose lambda(t) = sigma(t)^2 ( a standard choice from the literature), such
        that the score network and the target scores that are used are actually "sigma normalized" versions, ie,
        pre-multiplied by sigma.

        The loss that is computed is a Monte Carlo estimate of L, where we sample a mini-batch of relative position
        configurations {x0}; each of these configurations is noised with a random t value, with corresponding
        {sigma(t)} and {xt}.

        Args:
            batch : a dictionary that should contain a data sample.
            batch_idx :  index of the batch

        Returns:
            loss : the computed loss.
        """
        # The relative positions have dimensions [batch_size, number_of_atoms, spatial_dimension].
        assert "relative_positions" in batch, "The field 'relative_positions' is missing from the input."
        x0 = batch["relative_positions"]
        shape = x0.shape
        assert len(shape) == 3, (
            f"the shape of the relative_positions array should be [batch_size, number_of_atoms, spatial_dimensions]. "
            f"Got shape = {shape}."
        )
        batch_size = self._get_batch_size(batch)

        noise_sample = self.variance_sampler.get_random_noise_sample(batch_size)

        # noise_sample.sigma has  dimension [batch_size]. Broadcast these sigma values to be
        # of shape [batch_size, number_of_atoms, spatial_dimension], which can be interpreted
        # as [batch_size, (configuration)]. All the sigma values must be the same for a given configuration.
        sigmas = broadcast_batch_tensor_to_all_dimensions(
            batch_values=noise_sample.sigma, final_shape=shape
        )

        xt = self.noisy_position_sampler.get_noisy_position_sample(x0, sigmas)

        # The target is nabla log p_{t|0} (xt | x0): it is NOT the "score", but rather a "conditional" (on x0) score.
        target_normalized_conditional_scores = self._get_target_normalized_score(xt, x0, sigmas)

        predicted_normalized_scores = self._get_predicted_normalized_score(
            xt, noise_sample.time
        )

        loss = torch.nn.functional.mse_loss(
            predicted_normalized_scores, target_normalized_conditional_scores, reduction="mean"
        )

        output = dict(loss=loss,
                      real_relative_positions=x0,
                      noisy_relative_positions=xt,
                      sigmas=sigmas,
                      predicted_normalized_scores=predicted_normalized_scores.detach(),
                      target_normalized_conditional_scores=target_normalized_conditional_scores)
        return output

    def _get_target_normalized_score(
        self,
        noisy_relative_positions: torch.Tensor,
        real_relative_positions: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """Get target normalized score.

        It is assumed that the inputs are consistent, ie, the noisy relative positions correspond
        to the real relative positions noised with sigmas. It is also assumed that sigmas has
        been broadcast so that the same value sigma(t) is applied to all atoms + dimensions within a configuration.

        Args:
            noisy_relative_positions : noised relative positions.
                Tensor of dimensions [batch_size, number_of_atoms, spatial_dimension]
            real_relative_positions : original relative positions, before the addition of noise.
                Tensor of dimensions [batch_size, number_of_atoms, spatial_dimension]
            sigmas :
                Tensor of dimensions [batch_size, number_of_atoms, spatial_dimension]

        Returns:
        target normalized score: sigma times target score, ie, sigma times nabla_xt log P_{t|0}(xt| x0).
                Tensor of dimensions [batch_size, number_of_atoms, spatial_dimension]
        """
        delta_relative_positions = map_positions_to_unit_cell(noisy_relative_positions - real_relative_positions)
        target_normalized_scores = get_sigma_normalized_score(
            delta_relative_positions, sigmas, kmax=self.hyper_params.kmax_target_score
        )
        return target_normalized_scores

    def _get_predicted_normalized_score(
        self, noisy_relative_positions: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        """Get predicted normalized score.

        Args:
            noisy_relative_positions : noised relative positions.
                Tensor of dimensions [batch_size, number_of_atoms, spatial_dimension]
            time : Noise times for the noisy relative positions. It is assumed that the inputs are consistent, ie,
                the time values in this array correspond to the noise times used to create the noisy relative positions.
                Tensor of dimensions [batch_size].

        Returns:
            predicted normalized score: sigma times predicted score, ie, sigma times S_theta(xt, t).
                Tensor of dimensions [batch_size, number_of_atoms, spatial_dimension]
        """
        pos_key = self.sigma_normalized_score_network.position_key
        time_key = self.sigma_normalized_score_network.timestep_key
        augmented_batch = {
            pos_key: noisy_relative_positions,
            time_key: time.reshape(-1, 1),
        }
        predicted_normalized_scores = self.sigma_normalized_score_network(
            augmented_batch
        )
        return predicted_normalized_scores

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        output = self._generic_step(batch, batch_idx)
        loss = output["loss"]

        batch_size = self._get_batch_size(batch)

        # The 'train_step_loss' is only logged on_step, meaning it is a value for each batch
        self.log("train_step_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # The 'train_epoch_loss' is aggregated (batch_size weighted average) and logged once per epoch.
        self.log("train_epoch_loss", loss, batch_size=batch_size, on_step=False, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
        output = self._generic_step(batch, batch_idx)
        loss = output["loss"]
        batch_size = self._get_batch_size(batch)

        # The 'validation_epoch_loss' is aggregated (batch_size weighted average) and logged once per epoch.
        self.log("validation_epoch_loss", loss,
                 batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        return output

    def test_step(self, batch, batch_idx):
        """Runs a prediction step for testing, logging the loss."""
        output = self._generic_step(batch, batch_idx)
        loss = output["loss"]
        batch_size = self._get_batch_size(batch)
        # The 'test_epoch_loss' is aggregated (batch_size weighted average) and logged once per epoch.
        self.log("test_epoch_loss", loss, batch_size=batch_size, on_step=False, on_epoch=True)
        return output

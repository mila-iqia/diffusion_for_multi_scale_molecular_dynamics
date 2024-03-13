import logging
import typing

import pytorch_lightning as pl

from crystal_diffusion.models.optim import load_optimizer
from crystal_diffusion.models.score_network import BaseScoreNetwork

logger = logging.getLogger(__name__)


class PositionDiffusionLightningModel(pl.LightningModule):
    """Position Diffusion Lightning Model.

    This lightning model can train a score network predict the noise for relative positions.

    TODO : filling this class with what is needed to train a diffusion model.
    """

    def __init__(self, score_network: BaseScoreNetwork):
        """Init method.

        This initializes the class.

        Args:
            score_network: the model that computes the scores.
        """
        super().__init__()

        self.score_network = score_network

    def configure_optimizers(self):
        """Returns the combination of optimizer(s) and learning rate scheduler(s) to train with.

        Here, we read all the optimization-related hyperparameters from the config dictionary and
        create the required optimizer/scheduler combo.

        This function will be called automatically by the pytorch lightning trainer implementation.
        See https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html for more info
        on the expected returned elements.
        """
        # we use the generic loading function from the `model_loader` module, but it could be made
        # a direct part of the model (useful if we want layer-dynamic optimization)
        return load_optimizer(self.hparams, self)

    def _generic_step(
        self,
        batch: typing.Any,
        batch_idx: int,
    ) -> typing.Any:
        """Runs the prediction + evaluation step for training/validation/testing."""
        input_data, targets = batch
        preds = self(input_data)  # calls the forward pass of the model
        loss = self.loss_fn(preds, targets)
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

import dataclasses

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE


class HPLoggingCallback(Callback):
    """This callback is responsible for logging hyperparameters."""

    def on_train_start(self, trainer, pl_module):
        """Log hyperparameters when training starts."""
        assert hasattr(
            pl_module, "hyper_params"
        ), "The lightning module should have a hyper_params attribute for HP logging."
        hp_dict = dataclasses.asdict(pl_module.hyper_params)
        trainer.logger.log_hyperparams(hp_dict)


class TensorBoardDebuggingLoggingCallback(Callback):
    """Base class to log debugging information for plotting on TensorBoard."""

    def __init__(self):
        """Init method."""
        self.training_step_outputs = []

    @staticmethod
    def _get_tensorboard_logger(trainer):
        if type(trainer.logger) == TensorBoardLogger:
            return trainer.logger.experiment
        return None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Action to perform at the end of a training batch."""
        if self._get_tensorboard_logger(trainer) is None:
            return
        self.training_step_outputs.append(outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        """Action to perform at the end of a training epoch."""
        tbx_logger = self._get_tensorboard_logger(trainer)
        if tbx_logger is None:
            return

        if pl_module.global_step % trainer.log_every_n_steps == 0:
            self.log_artifact(pl_module, tbx_logger)
        # free up the memory
        self.training_step_outputs.clear()

    def log_artifact(self, pl_module, tbx_logger):
        """This method must create logging artifacts and log to the tbx logger."""
        raise NotImplementedError(
            "This method should be implemented to specific logging."
        )


class TensorboardHistogramLoggingCallback(TensorBoardDebuggingLoggingCallback):
    """This callback will log histograms of the predictions on tensorboard."""

    def log_artifact(self, pl_module, tbx_logger):
        """Create artifact and log to tensorboard."""
        targets = []
        predictions = []
        for output in self.training_step_outputs:
            targets.append(output["target_normalized_conditional_scores"].flatten())
            predictions.append(output["predicted_normalized_scores"].flatten())

        targets = torch.cat(targets)
        predictions = torch.cat(predictions)

        tbx_logger.add_histogram(
            "train/targets", targets, global_step=pl_module.global_step
        )
        tbx_logger.add_histogram(
            "train/predictions", predictions, global_step=pl_module.global_step
        )
        tbx_logger.add_histogram(
            "train/errors", targets - predictions, global_step=pl_module.global_step
        )


class TensorboardSamplesLoggingCallback(TensorBoardDebuggingLoggingCallback):
    """This callback will log histograms of the labels, predictions and errors on tensorboard."""

    def log_artifact(self, pl_module, tbx_logger):
        """Create artifact and log to tensorboard."""
        list_xt = []
        list_sigmas = []
        for output in self.training_step_outputs:
            list_xt.append(output["noisy_relative_positions"].flatten())
            list_sigmas.append(output["sigmas"].flatten())
        list_xt = torch.cat(list_xt)
        list_sigmas = torch.cat(list_sigmas)
        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        ax = fig.add_subplot(111)
        ax.set_title(f"Position Samples: global step = {pl_module.global_step}")
        ax.set_ylabel("$\\sigma$")
        ax.set_xlabel("position samples $x(t)$")
        ax.plot(list_xt, list_sigmas, "bo")
        tbx_logger.add_figure("train/samples", fig, global_step=pl_module.global_step)

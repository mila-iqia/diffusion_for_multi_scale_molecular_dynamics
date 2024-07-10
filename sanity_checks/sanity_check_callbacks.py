import torch
from matplotlib import pyplot as plt
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.generators.langevin_position_generator import \
    LangevinGenerator
from crystal_diffusion.models.score_networks.mlp_score_network import \
    MLPScoreNetworkParameters
from crystal_diffusion.namespace import NOISY_RELATIVE_COORDINATES
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.score.wrapped_gaussian_score import \
    get_sigma_normalized_score
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell

plt.style.use(PLOT_STYLE_PATH)


class TensorBoardDebuggingLoggingCallback(Callback):
    """Base class to log debugging information for plotting on TensorBoard."""

    def __init__(self):
        """Init method."""
        self.training_step_outputs = []

    @staticmethod
    def _get_tensorboard_logger(trainer):
        if type(trainer.logger) is TensorBoardLogger:
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


class TensorboardGeneratedSamplesLoggingCallback(TensorBoardDebuggingLoggingCallback):
    """This callback will log an image of a histogram of generated samples on tensorboard."""

    def __init__(self, noise_parameters: NoiseParameters,
                 number_of_corrector_steps: int,
                 score_network_parameters: MLPScoreNetworkParameters, number_of_samples: int):
        """Init method."""
        super().__init__()
        self.noise_parameters = noise_parameters
        self.number_of_corrector_steps = number_of_corrector_steps
        self.score_network_parameters = score_network_parameters
        self.number_of_atoms = score_network_parameters.number_of_atoms
        self.spatial_dimension = score_network_parameters.spatial_dimension
        self.number_of_samples = number_of_samples

    def log_artifact(self, pl_module, tbx_logger):
        """Create artifact and log to tensorboard."""
        sigma_normalized_score_network = pl_module.sigma_normalized_score_network
        pc_generator = LangevinGenerator(noise_parameters=self.noise_parameters,
                                         number_of_corrector_steps=self.number_of_corrector_steps,
                                         number_of_atoms=self.number_of_atoms,
                                         spatial_dimension=self.spatial_dimension,
                                         sigma_normalized_score_network=sigma_normalized_score_network)

        samples = pc_generator.sample(self.number_of_samples).flatten()

        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        ax = fig.add_subplot(111)
        ax.set_title(f"Generated Samples: global step = {pl_module.global_step}")
        ax.set_xlabel('$x$')
        ax.hist(samples, bins=101, range=(0, 1), label=f'{self.number_of_samples} samples')
        ax.set_title("Samples Count")
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([0., self.number_of_samples])
        fig.tight_layout()
        tbx_logger.add_figure("train/generated_samples", fig, global_step=pl_module.global_step)


class TensorboardScoreAndErrorLoggingCallback(TensorBoardDebuggingLoggingCallback):
    """This callback will log histograms of the labels, predictions and errors on tensorboard."""

    def __init__(self, x0: float):
        """Init method."""
        super().__init__()
        self.x0 = x0

    def log_artifact(self, pl_module, tbx_logger):
        """Create artifact and log to tensorboard."""
        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        fig.suptitle("Scores within 2 $\\sigma$ of Data")
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.set_ylabel('$\\sigma \\times S_{\\theta}(x, t)$')
        ax2.set_ylabel('$\\sigma \\times S_{\\theta}(x, t) - \\sigma \\nabla \\log P(x | 0)$')
        for ax in [ax1, ax2]:
            ax.set_xlabel('$x$')

        list_x = torch.linspace(0, 1, 1001)[:-1]

        times = torch.tensor([0.25, 0.75, 1.])
        sigmas = pl_module.variance_sampler._create_sigma_array(pl_module.variance_sampler.noise_parameters, times)

        with torch.no_grad():
            for time, sigma in zip(times, sigmas):
                times = time * torch.ones(1000).reshape(-1, 1)

                sigma_normalized_kernel = get_sigma_normalized_score(map_relative_coordinates_to_unit_cell(list_x
                                                                                                           - self.x0),
                                                                     sigma * torch.ones_like(list_x),
                                                                     kmax=4)
                predicted_normalized_scores = pl_module._get_predicted_normalized_score(list_x.reshape(-1, 1, 1),
                                                                                        times).flatten()

                error = predicted_normalized_scores - sigma_normalized_kernel

                # only plot the errors in the sampling region! These regions might be disconnected, let's make
                # sure the continuous lines make sense.
                mask1 = torch.abs(list_x - self.x0) < 2 * sigma
                mask2 = torch.abs(1. - list_x + self.x0) < 2 * sigma

                lines = ax1.plot(list_x[mask1], predicted_normalized_scores[mask1], lw=1, label='Prediction')
                color = lines[0].get_color()
                ax1.plot(list_x[mask2], predicted_normalized_scores[mask2], lw=1, color=color, label='_none_')

                ax1.plot(list_x[mask1], sigma_normalized_kernel[mask1], '--', lw=2, color=color, label='Target')
                ax1.plot(list_x[mask2], sigma_normalized_kernel[mask2], '--', lw=2, color=color, label='_none_')

                ax2.plot(list_x[mask1], error[mask1], '-', color=color,
                         label=f't = {time:4.3f}, $\\sigma$ = {sigma:4.3f}')
                ax2.plot(list_x[mask2], error[mask2], '-', color=color, label='_none_')

        for ax in [ax1, ax2]:
            ax.set_xlim([-0.05, 1.05])
            ax.legend(loc=3, prop={'size': 6})

        ax1.set_ylim([-3., 3.])
        ax2.set_ylim([-1., 1.])

        fig.tight_layout()

        tbx_logger.add_figure("train/scores", fig, global_step=pl_module.global_step)


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
            list_xt.append(output[NOISY_RELATIVE_COORDINATES].flatten())
            list_sigmas.append(output["sigmas"].flatten())
        list_xt = torch.cat(list_xt)
        list_sigmas = torch.cat(list_sigmas)
        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        ax = fig.add_subplot(111)
        ax.set_title(f"Position Samples: global step = {pl_module.global_step}")
        ax.set_ylabel("$\\sigma$")
        ax.set_xlabel("position samples $x(t)$")
        ax.plot(list_xt, list_sigmas, "bo")
        ax.set_xlim([-0.05, 1.05])
        fig.tight_layout()
        tbx_logger.add_figure("train/samples", fig, global_step=pl_module.global_step)

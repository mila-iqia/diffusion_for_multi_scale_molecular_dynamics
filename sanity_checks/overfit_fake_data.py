"""Overfit fake data.

A simple sanity check experiment to check the learning behavior of the position diffusion model.
The training data is taken to be a large batch of identical configurations composed of one atom in 1D at the origin.
This highly artificial case is useful to sanity check that the code behaves as expected:
 -  the loss should converge towards zero
 -  the trained score network should reproduce the perturbation kernel, at least in the regions where it is sampled.
"""
import os

import matplotlib.pyplot as plt
import pytorch_lightning
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from crystal_diffusion import ANALYSIS_RESULTS_DIR
from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.models.callbacks import (
    HPLoggingCallback, TensorboardHistogramLoggingCallback,
    TensorboardSamplesLoggingCallback)
from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                ValidOptimizerNames)
from crystal_diffusion.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from crystal_diffusion.models.score_network import MLPScoreNetworkParameters
from crystal_diffusion.samplers.predictor_corrector_position_sampler import \
    AnnealedLangevinDynamicsSampler
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.score.wrapped_gaussian_score import \
    get_sigma_normalized_score
from sanity_checks import SANITY_CHECK_FOLDER

plt.style.use(PLOT_STYLE_PATH)

batch_size = 4096
number_of_atoms = 1
spatial_dimension = 1
total_time_steps = 100
number_of_corrector_steps = 1

sigma_min = 0.005
sigma_max = 0.5

lr = 0.001
max_epochs = 3000

hidden_dimensions = [64, 128, 256]


score_network_parameters = MLPScoreNetworkParameters(
    number_of_atoms=number_of_atoms,
    hidden_dimensions=hidden_dimensions,
    spatial_dimension=spatial_dimension,
)

optimizer_parameters = OptimizerParameters(name=ValidOptimizerNames("adam"), learning_rate=lr)

noise_parameters = NoiseParameters(total_time_steps=total_time_steps, sigma_min=sigma_min, sigma_max=sigma_max)

hyper_params = PositionDiffusionParameters(
    score_network_parameters=score_network_parameters,
    optimizer_parameters=optimizer_parameters,
    noise_parameters=noise_parameters,
)


tbx_logger = TensorBoardLogger(save_dir=os.path.join(SANITY_CHECK_FOLDER, "tensorboard"), name="overfit_fake_data")

if __name__ == '__main__':

    pytorch_lightning.seed_everything(123)
    all_positions = torch.zeros(batch_size, number_of_atoms, spatial_dimension)
    data = [dict(relative_positions=configuration) for configuration in all_positions]
    train_dataloader = DataLoader(data, batch_size=batch_size)

    lightning_model = PositionDiffusionLightningModel(hyper_params)

    trainer = Trainer(accelerator='cpu',
                      max_epochs=max_epochs,
                      logger=tbx_logger,
                      log_every_n_steps=50,
                      callbacks=[HPLoggingCallback(),
                                 TensorboardHistogramLoggingCallback(),
                                 TensorboardSamplesLoggingCallback(),
                                 LearningRateMonitor(logging_interval='step')])
    trainer.fit(lightning_model, train_dataloaders=train_dataloader)

    sigma_normalized_score_network = lightning_model.sigma_normalized_score_network
    pc_sampler = AnnealedLangevinDynamicsSampler(noise_parameters=noise_parameters,
                                                 number_of_corrector_steps=number_of_corrector_steps,
                                                 number_of_atoms=number_of_atoms,
                                                 spatial_dimension=spatial_dimension,
                                                 sigma_normalized_score_network=sigma_normalized_score_network)
    n_samples = 1024
    samples = pc_sampler.sample(n_samples).flatten()

    fig_size = (1.5 * PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[1])
    fig = plt.figure(figsize=fig_size)
    fig.suptitle("Predictions, Targets and Errors within 2 $\\sigma$ of Data Point")
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.set_ylabel('$\\sigma \\times S_{\\theta}(x, t)$')
    ax2.set_ylabel('$\\sigma \\times S_{\\theta}(x, t) - \\sigma \\nabla \\log P(x | 0)$')
    ax3.set_ylabel('Counts')
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('$x$')

    ax3.hist(samples, bins=100, label=f'{n_samples} samples')
    ax3.set_title("Samples Count")

    list_x = torch.linspace(0, 1, 1001)[:-1]

    times = torch.tensor([0.25, 0.75, 1.])
    sigmas = lightning_model.variance_sampler._create_sigma_array(lightning_model.variance_sampler.noise_parameters,
                                                                  times)

    with torch.no_grad():
        for time, sigma in zip(times, sigmas):
            times = time * torch.ones(1000).reshape(-1, 1)

            sigma_normalized_kernel = get_sigma_normalized_score(list_x,
                                                                 sigma * torch.ones_like(list_x),
                                                                 kmax=4)
            predicted_normalized_scores = lightning_model._get_predicted_normalized_score(list_x.reshape(-1, 1, 1),
                                                                                          times).flatten()

            error = predicted_normalized_scores - sigma_normalized_kernel

            # only plot the errors in the sampling region!
            m1 = list_x < 2 * sigma
            m2 = 1. - list_x < 2 * sigma

            lines = ax1.plot(list_x[m1], predicted_normalized_scores[m1], lw=1, label='PREDICTION')
            color = lines[0].get_color()
            ax1.plot(list_x[m2], predicted_normalized_scores[m2], lw=1, color=color, label='_none_')

            ax1.plot(list_x[m1], sigma_normalized_kernel[m1], '--', lw=1, color=color, label='TARGET')
            ax1.plot(list_x[m2], sigma_normalized_kernel[m2], '--', lw=1, color=color, label='_none_')

            ax2.plot(list_x[m1], error[m1], '-', color=color,
                     label=f't = {time:4.3f}, $\\sigma$ = {sigma:4.3f}')
            ax2.plot(list_x[m2], error[m2], '-', color=color, label='_none_')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim([-0.05, 1.05])
        ax.legend(loc=0)

    fig.tight_layout()
    fig.savefig(ANALYSIS_RESULTS_DIR.joinpath("overfit_fake_data_trained_model_errors.png"))

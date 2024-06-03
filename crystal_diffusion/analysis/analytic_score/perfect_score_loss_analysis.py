import logging

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from crystal_diffusion import ANALYSIS_RESULTS_DIR
from crystal_diffusion.analysis import PLOT_STYLE_PATH
from crystal_diffusion.analysis.analytic_score.utils import get_exact_samples
from crystal_diffusion.callbacks.loss_monitoring_callback import \
    LossMonitoringCallback
from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                ValidOptimizerName)
from crystal_diffusion.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from crystal_diffusion.models.scheduler import (
    CosineAnnealingLRSchedulerParameters, ValidSchedulerName)
from crystal_diffusion.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from crystal_diffusion.namespace import RELATIVE_COORDINATES
from crystal_diffusion.samplers.noisy_relative_coordinates_sampler import \
    NoisyRelativeCoordinatesSampler
from crystal_diffusion.samplers.variance_sampler import (
    ExplodingVarianceSampler, NoiseParameters)

logger = logging.getLogger(__name__)


class AnalyticalScorePositionDiffusionLightningModel(PositionDiffusionLightningModel):
    """Analytical Score Position Diffusion Lightning Model.

    Overload the base class so we can properly feed in an analytical score network.
    This should not be in the main code as the analytical score is not a real model.
    """

    def __init__(self, hyper_params: PositionDiffusionParameters):
        """Init method.

        This initializes the class.
        """
        pl.LightningModule.__init__(self)

        self.hyper_params = hyper_params
        self.save_hyperparameters(logger=False)

        self.sigma_normalized_score_network = AnalyticalScoreNetwork(hyper_params.score_network_parameters)

        self.grads_are_needed_in_inference = True

        self.noisy_relative_coordinates_sampler = NoisyRelativeCoordinatesSampler()
        self.variance_sampler = ExplodingVarianceSampler(hyper_params.noise_parameters)


plt.style.use(PLOT_STYLE_PATH)

device = torch.device('cpu')

spatial_dimension = 3

number_of_atoms = 2

dataset_size = 100_000
batch_size = 1024

kmax = 1

spring_constant = 1000.

noise_parameters = NoiseParameters(total_time_steps=100,
                                   sigma_min=0.001,
                                   sigma_max=0.5)


# We will not optimize, so  this doesn't matter
dummy_optimizer_parameters = OptimizerParameters(name=ValidOptimizerName.adam, learning_rate=0.001, weight_decay=0.0)
dummy_scheduler_parameters = CosineAnnealingLRSchedulerParameters(name=ValidSchedulerName.cosine_annealing_lr, T_max=10)

loss_monitoring_callback = LossMonitoringCallback(number_of_bins=50,
                                                  sample_every_n_epochs=1,
                                                  spatial_dimension=spatial_dimension)

csv_logger = CSVLogger(save_dir=str(ANALYSIS_RESULTS_DIR / 'perfect_score_loss'), name=f"in_{spatial_dimension}D")


if __name__ == '__main__':

    box = torch.ones(spatial_dimension)

    equilibrium_relative_coordinates = torch.stack([0.25 * torch.ones(spatial_dimension),
                                                    0.75 * torch.ones(spatial_dimension)])
    inverse_covariance = torch.zeros(number_of_atoms, spatial_dimension, number_of_atoms, spatial_dimension)
    for atom_i in range(number_of_atoms):
        for alpha in range(spatial_dimension):
            inverse_covariance[atom_i, alpha, atom_i, alpha] = spring_constant

    # Create a dataloader
    exact_samples = get_exact_samples(equilibrium_relative_coordinates, inverse_covariance, dataset_size)

    dataset = [{RELATIVE_COORDINATES: x, 'box': box} for x in exact_samples]

    dataloader = DataLoader(dataset, batch_size=batch_size)

    score_network_parameters = AnalyticalScoreNetworkParameters(
        number_of_atoms=number_of_atoms,
        spatial_dimension=spatial_dimension,
        kmax=kmax,
        equilibrium_relative_coordinates=equilibrium_relative_coordinates,
        inverse_covariance=inverse_covariance)

    diffusion_params = PositionDiffusionParameters(
        score_network_parameters=score_network_parameters,
        optimizer_parameters=dummy_optimizer_parameters,
        scheduler_parameters=dummy_scheduler_parameters,
        noise_parameters=noise_parameters,
    )

    model = AnalyticalScorePositionDiffusionLightningModel(diffusion_params)

    trainer = pl.Trainer(callbacks=[loss_monitoring_callback],
                         max_epochs=1,
                         log_every_n_steps=10,
                         fast_dev_run=False,
                         logger=csv_logger,
                         inference_mode=False  # Do this or else the gradients don't work!
                         )

    # Run a validation epoch
    trainer.validate(model, dataloaders=[dataloader])

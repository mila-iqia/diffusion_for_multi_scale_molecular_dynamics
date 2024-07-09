import logging

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from crystal_diffusion import ANALYSIS_RESULTS_DIR
from crystal_diffusion.analysis import PLOT_STYLE_PATH
from crystal_diffusion.analysis.analytic_score.utils import (
    get_exact_samples, get_silicon_supercell)
from crystal_diffusion.callbacks.loss_monitoring_callback import \
    LossMonitoringCallback
from crystal_diffusion.models.loss import (MSELossParameters,
                                           create_loss_calculator)
from crystal_diffusion.models.optimizer import OptimizerParameters
from crystal_diffusion.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from crystal_diffusion.models.scheduler import \
    CosineAnnealingLRSchedulerParameters
from crystal_diffusion.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from crystal_diffusion.namespace import CARTESIAN_FORCES, RELATIVE_COORDINATES
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

        self.loss_calculator = create_loss_calculator(hyper_params.loss_parameters)
        self.noisy_relative_coordinates_sampler = NoisyRelativeCoordinatesSampler()
        self.variance_sampler = ExplodingVarianceSampler(hyper_params.noise_parameters)

    def on_validation_start(self) -> None:
        """On validation start."""
        # The analytical score is computed by explicitly calling autograd.
        torch.set_grad_enabled(True)


plt.style.use(PLOT_STYLE_PATH)

device = torch.device('cpu')


dataset_size = 100_000
batch_size = 1024

spatial_dimension = 3
kmax = 1
supercell_factor = 1
variance_parameter = 0.001 / supercell_factor

use_permutation_invariance = False

noise_parameters = NoiseParameters(total_time_steps=100,
                                   sigma_min=0.001,
                                   sigma_max=0.5)

# We will not optimize, so  this doesn't matter
dummy_optimizer_parameters = OptimizerParameters(name='adam', learning_rate=0.001, weight_decay=0.0)
dummy_scheduler_parameters = CosineAnnealingLRSchedulerParameters(T_max=10)

loss_monitoring_callback = LossMonitoringCallback(number_of_bins=50,
                                                  sample_every_n_epochs=1,
                                                  spatial_dimension=spatial_dimension)

csv_logger = CSVLogger(save_dir=str(ANALYSIS_RESULTS_DIR / 'perfect_score_loss'),
                       name=f"permutation_3_atoms_{use_permutation_invariance}")


if __name__ == '__main__':

    box = torch.ones(spatial_dimension)

    equilibrium_relative_coordinates = torch.from_numpy(
        get_silicon_supercell(supercell_factor=supercell_factor))

    number_of_atoms, _ = equilibrium_relative_coordinates.shape
    nd = number_of_atoms * spatial_dimension

    inverse_covariance = torch.diag(torch.ones(nd)) / variance_parameter
    inverse_covariance = inverse_covariance.reshape(number_of_atoms, spatial_dimension,
                                                    number_of_atoms, spatial_dimension)

    # Create a dataloader
    exact_samples = get_exact_samples(equilibrium_relative_coordinates, inverse_covariance, dataset_size)

    dataset = [{RELATIVE_COORDINATES: x, CARTESIAN_FORCES: torch.zeros_like(x), 'box': box} for x in exact_samples]

    dataloader = DataLoader(dataset, batch_size=batch_size)

    score_network_parameters = AnalyticalScoreNetworkParameters(
        number_of_atoms=number_of_atoms,
        spatial_dimension=spatial_dimension,
        use_permutation_invariance=use_permutation_invariance,
        kmax=kmax,
        equilibrium_relative_coordinates=equilibrium_relative_coordinates,
        variance_parameter=variance_parameter)

    diffusion_params = PositionDiffusionParameters(
        score_network_parameters=score_network_parameters,
        loss_parameters=MSELossParameters(),
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

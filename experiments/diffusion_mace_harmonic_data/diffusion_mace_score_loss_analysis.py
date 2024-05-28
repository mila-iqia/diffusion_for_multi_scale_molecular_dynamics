import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from crystal_diffusion.analysis import PLOT_STYLE_PATH
from crystal_diffusion.analysis.analytic_score.utils import (
    get_exact_samples, get_samples_harmonic_energy)
from crystal_diffusion.callbacks.callback_loader import create_all_callbacks
from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                ValidOptimizerName)
from crystal_diffusion.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from crystal_diffusion.models.scheduler import (
    ReduceLROnPlateauSchedulerParameters, ValidSchedulerName)
from crystal_diffusion.models.score_network import \
    DiffusionMACEScoreNetworkParameters
from crystal_diffusion.namespace import RELATIVE_COORDINATES
from crystal_diffusion.samplers.variance_sampler import NoiseParameters

logger = logging.getLogger(__name__)


plt.style.use(PLOT_STYLE_PATH)

spatial_dimension = 3
number_of_atoms = 2

dataset_size = 100_000
batch_size = 1024

spring_constant = 1000.

noise_parameters = NoiseParameters(total_time_steps=100,
                                   sigma_min=0.001,
                                   sigma_max=0.5)

score_network_parameters = DiffusionMACEScoreNetworkParameters(
    number_of_atoms=2,
    r_max=5.0,
    num_bessel=4,
    num_polynomial_cutoff=3,
    max_ell=1,
    interaction_cls="RealAgnosticResidualInteractionBlock",
    interaction_cls_first="RealAgnosticInteractionBlock",
    num_interactions=2,
    hidden_irreps="16x0e + 16x1o",
    MLP_irreps="16x0e",
    avg_num_neighbors=1,
    correlation=2,
    gate="silu",
    radial_MLP=[16, 16, 16],
    radial_type="bessel"
)

max_epochs = 100

acell = 5.5

# We will not optimize, so  this doesn't matter
optimizer_parameters = OptimizerParameters(name=ValidOptimizerName.adamw, learning_rate=0.001, weight_decay=0.0)
scheduler_parameters = ReduceLROnPlateauSchedulerParameters(name=ValidSchedulerName.reduce_lr_on_plateau,
                                                            factor=0.5,
                                                            patience=10)

loss_monitoring_parameters = dict(number_of_bins=50, sample_every_n_epochs=1, spatial_dimension=spatial_dimension)
diffusion_sampling_parameters = dict(noise=dict(total_time_steps=100,
                                                sigma_min=0.001,
                                                sigma_max=0.5),
                                     sampling=dict(spatial_dimension=3,
                                                   number_of_corrector_steps=1,
                                                   number_of_atoms=2,
                                                   number_of_samples=10,
                                                   sample_every_n_epochs=1,
                                                   record_samples=True,
                                                   cell_dimensions=[acell, acell, acell]
                                                   )
                                     )

callback_parameters = dict(loss_monitoring=loss_monitoring_parameters,
                           diffusion_sampling=diffusion_sampling_parameters)

output_directory = str(Path(__file__).parent / "output")

tensorboard_logger = TensorBoardLogger(save_dir=output_directory,
                                       default_hp_metric=False,
                                       name="tensorboard_logs",
                                       version=0,
                                       )

if __name__ == '__main__':

    box = acell * torch.ones(spatial_dimension)

    equilibrium_relative_coordinates = torch.stack([0.25 * torch.ones(spatial_dimension),
                                                    0.75 * torch.ones(spatial_dimension)])
    inverse_covariance = torch.zeros(number_of_atoms, spatial_dimension, number_of_atoms, spatial_dimension)
    for atom_i in range(number_of_atoms):
        for alpha in range(spatial_dimension):
            inverse_covariance[atom_i, alpha, atom_i, alpha] = spring_constant

    # Create a dataloader
    train_samples = get_exact_samples(equilibrium_relative_coordinates, inverse_covariance, dataset_size)

    train_energies = get_samples_harmonic_energy(equilibrium_relative_coordinates, inverse_covariance, train_samples)
    train_dataset = [{RELATIVE_COORDINATES: x, 'box': box, 'potential_energy': e}
                     for x, e in zip(train_samples, train_energies)]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    valid_samples = get_exact_samples(equilibrium_relative_coordinates, inverse_covariance, dataset_size // 10)
    valid_energies = get_samples_harmonic_energy(equilibrium_relative_coordinates, inverse_covariance, valid_samples)
    valid_dataset = [{RELATIVE_COORDINATES: x, 'box': box, 'potential_energy': e}
                     for x, e in zip(valid_samples, valid_energies)]
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    diffusion_params = PositionDiffusionParameters(
        score_network_parameters=score_network_parameters,
        optimizer_parameters=optimizer_parameters,
        scheduler_parameters=scheduler_parameters,
        noise_parameters=noise_parameters,
    )

    model = PositionDiffusionLightningModel(diffusion_params)

    callbacks_dict = create_all_callbacks(callback_parameters, output_directory=output_directory, verbose=True)

    trainer = pl.Trainer(callbacks=list(callbacks_dict.values()),
                         max_epochs=max_epochs,
                         log_every_n_steps=1,
                         fast_dev_run=False,
                         logger=tensorboard_logger,
                         )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

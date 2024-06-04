"""Overfit Diffusion MACE.

This script helps in the exploration of the Diffusion MACE architecture, trying different ideas
quickly to see if we can overfit the analytical score for a single example.
"""
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from torch.utils.data import DataLoader

from crystal_diffusion.analysis.analytic_score.utils import (get_exact_samples,
                                                             get_unit_cells)
from crystal_diffusion.callbacks.standard_callbacks import CustomProgressBar
from crystal_diffusion.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from crystal_diffusion.models.score_networks.diffusion_mace_score_network import (
    DiffusionMACEScoreNetwork, DiffusionMACEScoreNetworkParameters)
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)
from crystal_diffusion.samplers.noisy_relative_coordinates_sampler import \
    NoisyRelativeCoordinatesSampler
from crystal_diffusion.samplers.variance_sampler import (
    ExplodingVarianceSampler, NoiseParameters)
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from crystal_diffusion.utils.tensor_utils import \
    broadcast_batch_tensor_to_all_dimensions

torch.set_default_dtype(torch.float64)

run_id = 1

dim = 16
sigma_value = 0.01
number_of_mlp_layers = 1


experiment_name = f"DiffusionMace/run{run_id}"
root_dir = Path(__file__).parent / "overfitting_experiments"

output_directory = root_dir / experiment_name

tensorboard_logger = TensorBoardLogger(save_dir=str(output_directory),
                                       default_hp_metric=False,
                                       name=experiment_name,
                                       version=0,
                                       )


class DevDiffusionMaceLightningModel(pl.LightningModule):
    """This is a stub lightning module for the purpose of trying to overfit Diffusion Mace to the analytical score."""

    def __init__(self, diffusion_mace_score_network_parameters: DiffusionMACEScoreNetworkParameters):
        """Init method."""
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.sigma_normalized_score_network = DiffusionMACEScoreNetwork(diffusion_mace_score_network_parameters)

    def configure_optimizers(self):
        """Configure optimizers."""
        parameters_dict = dict(lr=0.01, weight_decay=0.0)
        optimizer = optim.AdamW(self.parameters(), **parameters_dict)
        return optimizer

    def training_step(self, batch, batch_idx):
        """Training step."""
        predicted_normalized_scores = self.sigma_normalized_score_network(batch)
        target_normalized_scores = batch['TARGET']
        loss = torch.nn.functional.mse_loss(predicted_normalized_scores, target_normalized_scores, reduction="mean")
        output = dict(loss=loss)
        self.log("train_step_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return output


spatial_dimension = 3
number_of_atoms = 2

dataset_size = 1
batch_size = dataset_size

diffusion_mace_score_network_parameters = DiffusionMACEScoreNetworkParameters(
    number_of_atoms=number_of_atoms,
    r_max=5.0,
    num_bessel=8,
    num_polynomial_cutoff=5,
    max_ell=2,
    interaction_cls="RealAgnosticResidualInteractionBlock",
    interaction_cls_first="RealAgnosticInteractionBlock",
    num_interactions=2,
    hidden_irreps=f"{dim}x0e + {dim}x1o + {dim}x2e",
    mlp_irreps=f"{dim}x0e",
    number_of_mlp_layers=number_of_mlp_layers,
    avg_num_neighbors=1,
    correlation=2,
    gate="silu",
    radial_MLP=[dim, dim, dim],
    radial_type="gaussian")


max_epochs = 1000
acell = 5.5

noisy_relative_coordinates_sampler = NoisyRelativeCoordinatesSampler()

noise_parameters = NoiseParameters(total_time_steps=100, sigma_min=0.001, sigma_max=0.5)
variance_sampler = ExplodingVarianceSampler(noise_parameters)


if __name__ == '__main__':
    torch.manual_seed(42)

    # ======================   Generate Harmonic potential samples ====================================

    spring_constant = 1000.
    box = acell * torch.ones(spatial_dimension)

    equilibrium_relative_coordinates = torch.stack([0.25 * torch.ones(spatial_dimension),
                                                    0.75 * torch.ones(spatial_dimension)])

    inverse_covariance = torch.zeros(number_of_atoms, spatial_dimension, number_of_atoms, spatial_dimension)
    for atom_i in range(number_of_atoms):
        for alpha in range(spatial_dimension):
            inverse_covariance[atom_i, alpha, atom_i, alpha] = spring_constant

    # ======================   Generate Random Data ====================================

    x0 = get_exact_samples(equilibrium_relative_coordinates, inverse_covariance, dataset_size)
    noise_sample = variance_sampler.get_random_noise_sample(dataset_size)
    sigmas = broadcast_batch_tensor_to_all_dimensions(batch_values=noise_sample.sigma, final_shape=x0.shape)
    sigmas = torch.ones_like(sigmas)

    xt = noisy_relative_coordinates_sampler.get_noisy_relative_coordinates_sample(x0, sigmas)

    cm = 0.5 * xt.sum(dim=1)
    xt = map_relative_coordinates_to_unit_cell(xt - cm)

    # ======================   Generate analytical scores ====================================

    analytical_score_network_parameters = AnalyticalScoreNetworkParameters(
        number_of_atoms=number_of_atoms,
        spatial_dimension=spatial_dimension,
        kmax=1,
        equilibrium_relative_coordinates=equilibrium_relative_coordinates,
        inverse_covariance=inverse_covariance)

    analytical_score_network = AnalyticalScoreNetwork(analytical_score_network_parameters)

    unit_cells = get_unit_cells(acell, spatial_dimension, number_of_samples=dataset_size)

    sigma = noise_sample.sigma.reshape(-1, 1)
    sigma = sigma_value * torch.ones_like(sigma)
    augmented_batch = {NOISY_RELATIVE_COORDINATES: xt,
                       TIME: noise_sample.time.reshape(-1, 1),
                       NOISE: sigma,
                       UNIT_CELL: unit_cells,
                       CARTESIAN_FORCES: torch.zeros_like(xt)}

    analytical_scores = analytical_score_network(augmented_batch)

    # ======================   Create a DataLoader ====================================
    train_dataset = []
    for idx in range(dataset_size):
        data = dict(TARGET=analytical_scores[idx])
        for key, value_array in augmented_batch.items():
            data[key] = value_array[idx]
        train_dataset.append(data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    # ======================   Train the Model ====================================
    model = DevDiffusionMaceLightningModel(diffusion_mace_score_network_parameters)

    callbacks = [CustomProgressBar()]

    trainer = pl.Trainer(callbacks=callbacks,
                         max_epochs=max_epochs,
                         log_every_n_steps=1,
                         fast_dev_run=False,
                         logger=tensorboard_logger,
                         )

    trainer.fit(model, train_dataloaders=train_dataloader)

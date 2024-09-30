import logging
from dataclasses import dataclass
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from torch import Tensor

from crystal_diffusion.generators.instantiate_generator import \
    instantiate_generator
from crystal_diffusion.metrics.kolmogorov_smirnov_metrics import \
    KolmogorovSmirnovMetrics
from crystal_diffusion.models.loss import (LossParameters,
                                           create_loss_calculator)
from crystal_diffusion.models.normalized_score_fokker_planck_error import (
    FokkerPlanckLossCalculator, FokkerPlankRegularizerParameters)
from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                load_optimizer)
from crystal_diffusion.models.scheduler import (SchedulerParameters,
                                                load_scheduler_dictionary)
from crystal_diffusion.models.score_networks.score_network import \
    ScoreNetworkParameters
from crystal_diffusion.models.score_networks.score_network_factory import \
    create_score_network
from crystal_diffusion.namespace import (CARTESIAN_FORCES, CARTESIAN_POSITIONS,
                                         NOISE, NOISY_RELATIVE_COORDINATES,
                                         RELATIVE_COORDINATES, TIME, UNIT_CELL)
from crystal_diffusion.oracle.energies import compute_oracle_energies
from crystal_diffusion.samplers.noisy_relative_coordinates_sampler import \
    NoisyRelativeCoordinatesSampler
from crystal_diffusion.samplers.variance_sampler import (
    ExplodingVarianceSampler, NoiseParameters)
from crystal_diffusion.samples.diffusion_sampling_parameters import \
    DiffusionSamplingParameters
from crystal_diffusion.samples.sampling import create_batch_of_samples
from crystal_diffusion.score.wrapped_gaussian_score import \
    get_sigma_normalized_score
from crystal_diffusion.utils.basis_transformations import (
    get_positions_from_coordinates, map_relative_coordinates_to_unit_cell)
from crystal_diffusion.utils.structure_utils import compute_distances_in_batch
from crystal_diffusion.utils.tensor_utils import \
    broadcast_batch_tensor_to_all_dimensions

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class PositionDiffusionParameters:
    """Position Diffusion parameters."""

    score_network_parameters: ScoreNetworkParameters
    loss_parameters: LossParameters
    optimizer_parameters: OptimizerParameters
    scheduler_parameters: Optional[SchedulerParameters] = None
    noise_parameters: NoiseParameters
    # convergence parameter for the Ewald-like sum of the perturbation kernel.
    kmax_target_score: int = 4
    diffusion_sampling_parameters: Optional[DiffusionSamplingParameters] = None


class PositionDiffusionLightningModel(pl.LightningModule):
    """Position Diffusion Lightning Model.

    This lightning model can train a score network predict the noise for relative coordinates.
    """

    def __init__(self, hyper_params: PositionDiffusionParameters):
        """Init method.

        This initializes the class.
        """
        super().__init__()

        self.hyper_params = hyper_params
        self.save_hyperparameters(
            logger=False
        )  # It is not the responsibility of this class to log its parameters.

        # we will model sigma x score
        self.sigma_normalized_score_network = create_score_network(
            hyper_params.score_network_parameters
        )

        self.loss_calculator = create_loss_calculator(hyper_params.loss_parameters)

        self.fokker_planck = hyper_params.loss_parameters.fokker_planck_weight != 0.0
        if self.fokker_planck:
            fokker_planck_parameters = FokkerPlankRegularizerParameters(
                weight=hyper_params.loss_parameters.fokker_planck_weight)
            self.fokker_planck_loss_calculator = FokkerPlanckLossCalculator(self.sigma_normalized_score_network,
                                                                            hyper_params.noise_parameters,
                                                                            fokker_planck_parameters)

        self.noisy_relative_coordinates_sampler = NoisyRelativeCoordinatesSampler()
        self.variance_sampler = ExplodingVarianceSampler(hyper_params.noise_parameters)

        self.generator = None
        self.structure_ks_metric = None
        self.energy_ks_metric = None

        self.draw_samples = hyper_params.diffusion_sampling_parameters is not None
        if self.draw_samples:
            self.metrics_parameters = (
                self.hyper_params.diffusion_sampling_parameters.metrics_parameters
            )
            if self.metrics_parameters.compute_structure_factor:
                self.structure_ks_metric = KolmogorovSmirnovMetrics()
            if self.metrics_parameters.compute_energies:
                self.energy_ks_metric = KolmogorovSmirnovMetrics()

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
            scheduler_dict = load_scheduler_dictionary(
                scheduler_parameters=self.hyper_params.scheduler_parameters,
                optimizer=optimizer,
            )
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
        # The RELATIVE_COORDINATES have dimensions [batch_size, number_of_atoms, spatial_dimension].
        assert (
            RELATIVE_COORDINATES in batch
        ), f"The field '{RELATIVE_COORDINATES}' is missing from the input."
        batch_size = batch[RELATIVE_COORDINATES].shape[0]
        return batch_size

    def _generic_step(
        self,
        batch: Any,
        batch_idx: int,
        no_conditional: bool = False,
    ) -> Any:
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

        The loss that is computed is a Monte Carlo estimate of L, where we sample a mini-batch of relative coordinates
        configurations {x0}; each of these configurations is noised with a random t value, with corresponding
        {sigma(t)} and {xt}.

        Args:
            batch : a dictionary that should contain a data sample.
            batch_idx :  index of the batch
            no_conditional (optional): if True, do not use the conditional option of the forward. Used for validation.

        Returns:
            loss : the computed loss.
        """
        # The RELATIVE_COORDINATES have dimensions [batch_size, number_of_atoms, spatial_dimension].
        assert (
            RELATIVE_COORDINATES in batch
        ), f"The field '{RELATIVE_COORDINATES}' is missing from the input."
        x0 = batch[RELATIVE_COORDINATES]
        shape = x0.shape
        assert len(shape) == 3, (
            f"the shape of the RELATIVE_COORDINATES array should be [batch_size, number_of_atoms, spatial_dimensions]. "
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

        xt = self.noisy_relative_coordinates_sampler.get_noisy_relative_coordinates_sample(
            x0, sigmas
        )

        # The target is nabla log p_{t|0} (xt | x0): it is NOT the "score", but rather a "conditional" (on x0) score.
        target_normalized_conditional_scores = self._get_target_normalized_score(
            xt, x0, sigmas
        )

        unit_cell = torch.diag_embed(
            batch["box"]
        )  # from (batch, spatial_dim) to (batch, spatial_dim, spatial_dim)

        forces = batch[CARTESIAN_FORCES]

        augmented_batch = {
            NOISY_RELATIVE_COORDINATES: xt,
            TIME: noise_sample.time.reshape(-1, 1),
            NOISE: noise_sample.sigma.reshape(-1, 1),
            UNIT_CELL: unit_cell,
            CARTESIAN_FORCES: forces,
        }

        use_conditional = None if no_conditional is False else False
        predicted_normalized_scores = self.sigma_normalized_score_network(
            augmented_batch, conditional=use_conditional
        )

        unreduced_loss = self.loss_calculator.calculate_unreduced_loss(
            predicted_normalized_scores,
            target_normalized_conditional_scores,
            sigmas,
        )
        loss = torch.mean(unreduced_loss)

        output = dict(
            raw_loss=loss.detach(),
            unreduced_loss=unreduced_loss.detach(),
            sigmas=sigmas,
            predicted_normalized_scores=predicted_normalized_scores.detach(),
            target_normalized_conditional_scores=target_normalized_conditional_scores,
        )
        output[RELATIVE_COORDINATES] = x0
        output[NOISY_RELATIVE_COORDINATES] = xt

        if self.fokker_planck:
            logger.info(f"          * Computing Fokker-Planck loss term for {batch_idx}")
            fokker_planck_loss = self.fokker_planck_loss_calculator.compute_fokker_planck_loss_term(augmented_batch)
            logger.info(f"            Done Computing Fokker-Planck loss term for {batch_idx}")

            output['fokker_planck_loss'] = fokker_planck_loss.detach()
            output['loss'] = loss + fokker_planck_loss
        else:
            output['loss'] = loss

        return output

    def _get_target_normalized_score(
        self,
        noisy_relative_coordinates: torch.Tensor,
        real_relative_coordinates: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """Get target normalized score.

        It is assumed that the inputs are consistent, ie, the noisy relative coordinates correspond
        to the real relative coordinates noised with sigmas. It is also assumed that sigmas has
        been broadcast so that the same value sigma(t) is applied to all atoms + dimensions within a configuration.

        Args:
            noisy_relative_coordinates : noised relative coordinates.
                Tensor of dimensions [batch_size, number_of_atoms, spatial_dimension]
            real_relative_coordinates : original relative coordinates, before the addition of noise.
                Tensor of dimensions [batch_size, number_of_atoms, spatial_dimension]
            sigmas :
                Tensor of dimensions [batch_size, number_of_atoms, spatial_dimension]

        Returns:
        target normalized score: sigma times target score, ie, sigma times nabla_xt log P_{t|0}(xt| x0).
                Tensor of dimensions [batch_size, number_of_atoms, spatial_dimension]
        """
        delta_relative_coordinates = map_relative_coordinates_to_unit_cell(
            noisy_relative_coordinates - real_relative_coordinates
        )
        target_normalized_scores = get_sigma_normalized_score(
            delta_relative_coordinates, sigmas, kmax=self.hyper_params.kmax_target_score
        )
        return target_normalized_scores

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        output = self._generic_step(batch, batch_idx)
        loss = output["loss"]

        batch_size = self._get_batch_size(batch)

        # The 'train_step_loss' is only logged on_step, meaning it is a value for each batch
        self.log("train_step_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # The 'train_epoch_loss' is aggregated (batch_size weighted average) and logged once per epoch.
        self.log(
            "train_epoch_loss",
            loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )

        if self.fokker_planck:
            self.log("train_step_fokker_planck_loss", output['fokker_planck_loss'],
                     on_step=True, on_epoch=False, prog_bar=True)
            self.log(
                "train_epoch_fokker_planck_loss",
                output['fokker_planck_loss'],
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
            self.log("train_step_raw_loss", output['raw_loss'],
                     on_step=True, on_epoch=False, prog_bar=True)
            self.log(
                "train_epoch_raw_loss",
                output['raw_loss'],
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

        logger.info(f"         Done training step with batch index {batch_idx}")
        return output

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        """Backward method."""
        if self.fokker_planck:
            loss.backward(retain_graph=True)
        else:
            super().backward(loss, *args, **kwargs)

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
        logger.info(f"  - Starting validation step with batch index {batch_idx}")
        output = self._generic_step(batch, batch_idx, no_conditional=True)
        loss = output["loss"]
        batch_size = self._get_batch_size(batch)

        # The 'validation_epoch_loss' is aggregated (batch_size weighted average) and logged once per epoch.
        self.log(
            "validation_epoch_loss",
            loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.fokker_planck:
            self.log(
                "validation_epoch_fokker_planck_loss",
                output['fokker_planck_loss'],
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "validation_epoch_raw_loss",
                output['raw_loss'],
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

        if not self.draw_samples:
            return output

        if self.metrics_parameters.compute_energies:
            reference_energies = batch["potential_energy"]
            self.energy_ks_metric.register_reference_samples(reference_energies.cpu())

        if self.metrics_parameters.compute_structure_factor:
            basis_vectors = torch.diag_embed(batch["box"])
            cartesian_positions = get_positions_from_coordinates(
                relative_coordinates=batch[RELATIVE_COORDINATES],
                basis_vectors=basis_vectors,
            )

            reference_distances = compute_distances_in_batch(
                cartesian_positions=cartesian_positions,
                unit_cell=basis_vectors,
                max_distance=self.metrics_parameters.structure_factor_max_distance,
            )
            self.structure_ks_metric.register_reference_samples(
                reference_distances.cpu()
            )

        return output

    def test_step(self, batch, batch_idx):
        """Runs a prediction step for testing, logging the loss."""
        output = self._generic_step(batch, batch_idx)
        loss = output["loss"]
        batch_size = self._get_batch_size(batch)

        # The 'test_epoch_loss' is aggregated (batch_size weighted average) and logged once per epoch.
        self.log(
            "test_epoch_loss", loss, batch_size=batch_size, on_step=False, on_epoch=True
        )
        if self.fokker_planck:
            self.log(
                "test_epoch_fokker_planck_loss",
                output['fokker_planck_loss'],
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "test_epoch_raw_loss",
                output['raw_loss'],
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

        return output

    def generate_samples(self):
        """Generate a batch of samples."""
        assert (
            self.hyper_params.diffusion_sampling_parameters is not None
        ), "sampling parameters must be provided to create a generator."
        with torch.no_grad():
            logger.info("Creating Generator for sampling")
            self.generator = instantiate_generator(
                sampling_parameters=self.hyper_params.diffusion_sampling_parameters.sampling_parameters,
                noise_parameters=self.hyper_params.diffusion_sampling_parameters.noise_parameters,
                sigma_normalized_score_network=self.sigma_normalized_score_network,
            )
            logger.info(f"Generator type : {type(self.generator)}")

            logger.info("       * Drawing samples")
            samples_batch = create_batch_of_samples(
                generator=self.generator,
                sampling_parameters=self.hyper_params.diffusion_sampling_parameters.sampling_parameters,
                device=self.device,
            )
            logger.info("         Done drawing samples")
        return samples_batch

    def on_validation_epoch_end(self) -> None:
        """On validation epoch end."""
        if not self.draw_samples:
            return

        logger.info("Drawing samples at the end of the validation epoch.")
        samples_batch = self.generate_samples()

        if self.metrics_parameters.compute_energies:
            logger.info("   * Computing sample energies")
            sample_energies = compute_oracle_energies(samples_batch)
            logger.info("   * Registering sample energies")
            self.energy_ks_metric.register_predicted_samples(sample_energies.cpu())

            (
                ks_distance,
                p_value,
            ) = self.energy_ks_metric.compute_kolmogorov_smirnov_distance_and_pvalue()
            self.log(
                "validation_ks_distance_energy",
                ks_distance,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "validation_ks_p_value_energy", p_value, on_step=False, on_epoch=True
            )
            logger.info("   * Done logging sample energies")

        if self.metrics_parameters.compute_structure_factor:
            logger.info("   * Computing sample distances")
            sample_distances = compute_distances_in_batch(
                cartesian_positions=samples_batch[CARTESIAN_POSITIONS],
                unit_cell=samples_batch[UNIT_CELL],
                max_distance=self.metrics_parameters.structure_factor_max_distance,
            )

            logger.info("   * Registering sample distances")
            self.structure_ks_metric.register_predicted_samples(sample_distances.cpu())

            (
                ks_distance,
                p_value,
            ) = (
                self.structure_ks_metric.compute_kolmogorov_smirnov_distance_and_pvalue()
            )
            self.log(
                "validation_ks_distance_structure",
                ks_distance,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "validation_ks_p_value_structure", p_value, on_step=False, on_epoch=True
            )
            logger.info("   * Done logging sample distances")

    def on_validation_start(self) -> None:
        """On validation start."""
        logger.info("Clearing generator and metrics on validation start.")
        # Clear out any dangling state.
        self.generator = None
        if self.metrics_parameters.compute_energies:
            self.energy_ks_metric.reset()

        if self.metrics_parameters.compute_structure_factor:
            self.structure_ks_metric.reset()

    def on_train_start(self) -> None:
        """On train start."""
        logger.info("Clearing generator and metrics on train start.")
        # Clear out any dangling state.
        self.generator = None
        if self.metrics_parameters.compute_energies:
            self.energy_ks_metric.reset()

        if self.metrics_parameters.compute_structure_factor:
            self.structure_ks_metric.reset()

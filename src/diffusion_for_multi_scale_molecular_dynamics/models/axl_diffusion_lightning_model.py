import logging
from dataclasses import dataclass
from typing import Any, Optional

import einops
import pytorch_lightning as pl
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.instantiate_generator import \
    instantiate_generator
from diffusion_for_multi_scale_molecular_dynamics.generators.trajectory_initializer import \
    instantiate_trajectory_initializer
from diffusion_for_multi_scale_molecular_dynamics.loss import \
    create_loss_calculator
from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import \
    LossParameters
from diffusion_for_multi_scale_molecular_dynamics.metrics.kolmogorov_smirnov_metrics import \
    KolmogorovSmirnovMetrics
from diffusion_for_multi_scale_molecular_dynamics.models.optimizer import (
    OptimizerParameters, check_if_optimizer_is_none, load_optimizer)
from diffusion_for_multi_scale_molecular_dynamics.models.scheduler import (
    SchedulerParameters, load_scheduler_dictionary)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network_factory import \
    create_score_network
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, AXL, AXL_COMPOSITION, AXL_NAME_DICT, CARTESIAN_FORCES,
    CARTESIAN_POSITIONS, LATTICE_PARAMETERS, NOISE, NOISY_ATOM_TYPES,
    NOISY_AXL_COMPOSITION, NOISY_LATTICE_PARAMETERS,
    NOISY_RELATIVE_COORDINATES, Q_BAR_MATRICES, Q_BAR_TM1_MATRICES, Q_MATRICES,
    RELATIVE_COORDINATES, TIME, TIME_INDICES, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from diffusion_for_multi_scale_molecular_dynamics.noisers.atom_types_noiser import \
    AtomTypesNoiser
from diffusion_for_multi_scale_molecular_dynamics.noisers.lattice_noiser import \
    LatticeDataParameters, LatticeNoiser
from diffusion_for_multi_scale_molecular_dynamics.noisers.relative_coordinates_noiser import \
    RelativeCoordinatesNoiser
from diffusion_for_multi_scale_molecular_dynamics.oracle.energy_oracle import \
    OracleParameters
from diffusion_for_multi_scale_molecular_dynamics.oracle.energy_oracle_factory import \
    create_energy_oracle
from diffusion_for_multi_scale_molecular_dynamics.regularizers.regularizer import \
    RegularizerParameters
from diffusion_for_multi_scale_molecular_dynamics.regularizers.regularizer_factory import \
    create_regularizer
from diffusion_for_multi_scale_molecular_dynamics.sampling.diffusion_sampling import \
    create_batch_of_samples
from diffusion_for_multi_scale_molecular_dynamics.sampling.diffusion_sampling_parameters import \
    DiffusionSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.score.gaussian_score import get_lattice_sigma_normalized_score
from diffusion_for_multi_scale_molecular_dynamics.score.wrapped_gaussian_score import \
    get_coordinates_sigma_normalized_score
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates, map_relative_coordinates_to_unit_cell)
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    class_index_to_onehot
from diffusion_for_multi_scale_molecular_dynamics.utils.noise_utils import \
    scale_sigma_by_number_of_atoms
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import \
    compute_distances_in_batch
from diffusion_for_multi_scale_molecular_dynamics.utils.tensor_utils import broadcast_batch_tensor_to_all_dimensions

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AXLDiffusionParameters:
    """AXL (atom, relative coordinates, lattice) Diffusion parameters."""

    score_network_parameters: ScoreNetworkParameters
    loss_parameters: LossParameters
    optimizer_parameters: OptimizerParameters
    scheduler_parameters: Optional[SchedulerParameters] = None
    # convergence parameter for the Ewald-like sum of the perturbation kernel for coordinates.
    kmax_target_score: int = 4
    regularizer_parameters: Optional[RegularizerParameters] = None
    diffusion_sampling_parameters: Optional[DiffusionSamplingParameters] = None
    oracle_parameters: Optional[OracleParameters] = None
    lattice_parameters: LatticeDataParameters


class AXLDiffusionLightningModel(pl.LightningModule):
    """AXL Diffusion Lightning Model.

    This lightning model can train a score network to predict the noise for relative coordinates, atom types and lattice
    vectors.
    """

    def __init__(self, hyper_params: AXLDiffusionParameters):
        """Init method.

        This initializes the class.
        """
        super().__init__()

        self.hyper_params = hyper_params
        self.num_atom_types = hyper_params.score_network_parameters.num_atom_types
        self.save_hyperparameters(
            logger=False
        )  # It is not the responsibility of this class to log its parameters.

        if check_if_optimizer_is_none(self.hyper_params.optimizer_parameters):
            # If the config indicates None as the optimizer, then no optimization should
            # take place.
            self.automatic_optimization = False

        # the score network is expected to produce an output as an AXL namedtuple:
        # atom: unnormalized estimate of p(a_0 | a_t)
        # relative coordinates: estimate of \sigma \nabla_{x_t} p_{t|0}(x_t | x_0)
        # lattices: TODO
        self.axl_network = create_score_network(hyper_params.score_network_parameters)

        # loss is an AXL object with one loss for each element (atom type, coordinate, lattice)
        self.loss_calculator = create_loss_calculator(hyper_params.loss_parameters)

        self.loss_weights = AXL(A=hyper_params.loss_parameters.atom_types_lambda_weight,
                                X=hyper_params.loss_parameters.relative_coordinates_lambda_weight,
                                L=hyper_params.loss_parameters.lattice_lambda_weight)

        # noisy samplers for atom types, coordinates and lattice vectors
        self.noisers = AXL(
            A=AtomTypesNoiser(),
            X=RelativeCoordinatesNoiser(),
            L=LatticeNoiser(hyper_params.lattice_parameters),
        )

        self.noise_scheduler = NoiseScheduler(
            hyper_params.noise_parameters,
            num_classes=self.num_atom_types + 1,  # add 1 for the MASK class
        )

        self.generator = None
        self.structure_ks_metric = None
        self.energy_ks_metric = None
        self.oracle = None
        self.regularizer = None

        if hyper_params.regularizer_parameters is not None:
            self.regularizer = create_regularizer(hyper_params.regularizer_parameters)

        self.draw_samples = hyper_params.diffusion_sampling_parameters is not None
        if self.draw_samples:
            self.metrics_parameters = (
                self.hyper_params.diffusion_sampling_parameters.metrics_parameters
            )
            if self.metrics_parameters.compute_structure_factor:
                self.structure_ks_metric = KolmogorovSmirnovMetrics()
            if self.metrics_parameters.compute_energies:
                self.energy_ks_metric = KolmogorovSmirnovMetrics()
                assert self.hyper_params.oracle_parameters is not None, \
                    "Energies cannot be computed without a configured energy oracle."
                self.oracle = create_energy_oracle(self.hyper_params.oracle_parameters)

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
        r"""Generic step.

        This "generic step" computes the loss for any of the possible lightning "steps".

        The loss is defined as a sum of 3 components:

        .. math::
            L = L_x + L_a + L_L

        where :math:`L_x` is the loss for the coordinate diffusion, :math:`L_a` for the atom type diffusion and
        :math:`L_L` for the lattice.

        The loss for the coordinate diffusion is defined as:

        .. math::
            L_x = 1 / T \int_0^T dt \lambda(t) E_{x0 ~ p_data} E_{xt~ p_{t| 0}}
                    [|S_\theta(xt, t) - \nabla_{xt} \log p_{t | 0} (xt | x0)|^2]

        Where
                        :math:`T`   : time range of the noising process
                 :math:`S_\theta`   : score network
                  :math:`p_{t|0}`   : perturbation kernel
            :math:`\nabla \log p`   : the target score
               :math:`\lambda(t)`   : is arbitrary, but chosen for convenience.

        In this implementation, we choose :math:`\lambda(t) = \sigma(t)^2` (a standard choice from the literature), such
        that the score network and the target scores that are used are actually "sigma normalized" versions, ie,
        pre-multiplied by sigma.

        For the atom type diffusion, the loss is defined as:

        .. math::
            L_a = E_{a_0 ~ p_\textrm{data}} [ \sum_{t=2}^T E_{a_t ~ p_{t|0}
                [D_{KL}[q(a_{t-1} | a_t, a_0) || p_theta(a_{t-1} | a_{t}) - \lambda_CE log p_\theta(a_0 | a_t)]
                - E_{a_1 ~ p_{t=1|0}} log p_\theta(a_0 | a_1) ]

        The loss that is computed is a Monte Carlo estimate of L, where we sample a mini-batch of relative coordinates
        configurations {x0} and atom types {a_0}; each of these configurations is noised with a random t value,
        with corresponding {sigma(t)}, {xt}, {beta(t)} and {a(t)}. Note the :math:`beta(t)` is used to compute the true
        posterior :math:`q(a_{t-1} | a_t, a_0)` and :math:`p_\theta(a_{t-1} | a_t)` in the atom type loss.

        Args:
            batch : a dictionary that should contain a data sample.
            batch_idx : index of the batch
            no_conditional (optional): if True, do not use the conditional option of the forward. Used for validation.

        Returns:
            output_dictionary : contains the loss, the predictions and various other useful tensors.
        """
        a0 = batch[ATOM_TYPES]
        x0 = batch[RELATIVE_COORDINATES]
        l0 = batch[LATTICE_PARAMETERS]
        composition = AXL(A=a0, X=x0, L=l0)

        at = batch[NOISY_ATOM_TYPES]
        xt = batch[NOISY_RELATIVE_COORDINATES]
        lt = batch[NOISY_LATTICE_PARAMETERS]
        noisy_composition = AXL(A=at, X=xt, L=lt)

        # Get the loss targets
        # Coordinates: The target is :math:`sigma(t) \nabla  log p_{t|0} (xt | x0)`
        # it is NOT the "score", but rather a "conditional" (on x0) score.
        sigmas = einops.repeat(batch[NOISE],
                               "batch 1 -> batch natoms space",
                               natoms=x0.shape[1], space=x0.shape[2])
        target_coordinates_normalized_conditional_scores = (
            self._get_coordinates_target_normalized_score(xt, x0, sigmas)
        )

        # for the atom types, the loss is constructed from the Q and Qbar matrices

        # Lattice: he target is :math:`sigma(t) \nabla  log p_{t|0} (lt | l0)`
        # it is NOT the "score", but rather a "conditional" (on l0) score.
        sigmas_for_lattice = einops.repeat(batch[NOISE],
                                           "batch 1 -> batch space",
                                           space=l0.shape[-1])
        # same values as for X diffusion, but different shape
        num_atoms = (
                torch.ones_like(l0) * a0.shape[1]
        )  # TODO should depend on data - not a constant
        # num_atoms should be broadcasted to match sigmas_for_lattice
        sigmas_n = scale_sigma_by_number_of_atoms(
            sigmas_for_lattice, num_atoms, spatial_dimension=l0.shape[-1]
        )
        target_lattice_normalized_conditional_scores = (
            self._get_lattice_target_normalized_score(lt, l0, sigmas_n)
        )

        augmented_batch = {
            NOISY_AXL_COMPOSITION: noisy_composition,
            TIME: batch[TIME],
            NOISE: batch[NOISE],
            CARTESIAN_FORCES: batch[CARTESIAN_FORCES],
        }

        use_conditional = None if no_conditional is False else False

        # this output is expected to be an AXL object
        model_predictions = self.axl_network(
            augmented_batch, conditional=use_conditional
        )
        # X score network output: an estimate of the sigma normalized score for the coordinates,
        # A score network output: an unnormalized estimate of p(a_0 | a_t) for the atom types
        # L score network output: an estimate of the sigma normalized score for the lattice parameters

        unreduced_loss_coordinates = self.loss_calculator.X.calculate_unreduced_loss(
            model_predictions.X,
            target_coordinates_normalized_conditional_scores,
            sigmas,
        )

        # we also need the atom types to be one-hot vector and not a class index
        a0_onehot = class_index_to_onehot(a0, self.num_atom_types + 1)
        at_onehot = class_index_to_onehot(at, self.num_atom_types + 1)

        unreduced_loss_atom_types = self.loss_calculator.A.calculate_unreduced_loss(
            predicted_logits=model_predictions.A,
            one_hot_real_atom_types=a0_onehot,
            one_hot_noisy_atom_types=at_onehot,
            time_indices=batch[TIME_INDICES],
            q_matrices=batch[Q_MATRICES],
            q_bar_matrices=batch[Q_BAR_MATRICES],
            q_bar_tm1_matrices=batch[Q_BAR_TM1_MATRICES],
        )

        unreduced_loss_lattice = self.loss_calculator.L.calculate_unreduced_loss(
            model_predictions.L,
            target_lattice_normalized_conditional_scores,
            sigmas_for_lattice,
        )

        aggregated_weighted_loss = (
            self.loss_weights.X * unreduced_loss_coordinates.mean(
                dim=(-2, -1)
            )  # batch, num_atoms, spatial_dimension
            + self.loss_weights.L * unreduced_loss_lattice.mean(dim=-1)
            + self.loss_weights.A * unreduced_loss_atom_types.mean(dim=(-2, -1))  # batch, num_atoms, num_atom_types
        )

        weighted_loss = torch.mean(aggregated_weighted_loss)

        unreduced_loss = AXL(
            A=unreduced_loss_atom_types.detach(),
            X=unreduced_loss_coordinates.detach(),
            L=unreduced_loss_lattice.detach(),
        )

        model_predictions_detached = AXL(
            A=model_predictions.A.detach(),
            X=model_predictions.X.detach(),
            L=model_predictions.L.detach(),
        )

        output = dict(
            unreduced_loss=unreduced_loss,
            loss=weighted_loss,
            sigmas=sigmas,
            model_predictions=model_predictions_detached,
            target_coordinates_normalized_conditional_scores=target_coordinates_normalized_conditional_scores,
            target_lattice_normalized_conditional_scores=target_lattice_normalized_conditional_scores,
        )

        if self.regularizer and self.regularizer.can_regularizer_run():
            # Use the same times and atom types as in the noised composition. Random
            # relative coordinates will be drawn internally.
            weighted_regularizer_loss = (
                self.regularizer.compute_weighted_regularizer_loss(
                    score_network=self.axl_network,
                    augmented_batch=augmented_batch,
                    current_epoch=self.current_epoch))

            output["loss"] += weighted_regularizer_loss
            output["regularizer_loss"] = weighted_regularizer_loss

        output[AXL_COMPOSITION] = composition
        output[NOISY_AXL_COMPOSITION] = noisy_composition
        output[TIME] = augmented_batch[TIME]

        return output

    def _get_coordinates_target_normalized_score(
        self,
        noisy_relative_coordinates: torch.Tensor,
        real_relative_coordinates: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """Get target normalized score for the relative coordinates.

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
        target_normalized_scores = get_coordinates_sigma_normalized_score(
            delta_relative_coordinates, sigmas, kmax=self.hyper_params.kmax_target_score
        )
        return target_normalized_scores

    def _get_lattice_target_normalized_score(
        self,
        noisy_lattice_parameters: torch.Tensor,
        real_lattice_parameters: torch.Tensor,
        sigmas_n: torch.Tensor,
    ) -> torch.Tensor:
        """Get target normalized score for the lattice parameters.

        It is assumed that the inputs are consistent, ie, the noisy lattice parameters correspond
        to the real lattice parameters noised with sigmas and betas (related to alpha bars). It is also assumed that
        sigmas have been broadcast so that the same value sigma(t) is applied to all lattice parameters within a
        configuration.

        Args:
            noisy_lattice_parameters : noised lattice parameters.
                Tensor of dimensions [batch_size, spatial_dimension * (spatial_dimension + 1) / 2]
            real_lattice_parameters : original lattice coordinates, before the addition of noise.
                Tensor of dimensions [batch_size, spatial_dimension * (spatial_dimension + 1) / 2]
            sigmas_n : variance scaled by the number of atoms
                Tensor of dimensions [batch_size, spatial_dimension * (spatial_dimension + 1) / 2]

        Returns:
            target normalized score: sigma times target score, ie, sigma times nabla_lt log P_{t|0}(lt| l0).
                Tensor of dimensions [batch_size, spatial_dimension * (spatial_dimension + 1) / 2]
        """
        target_normalized_scores = get_lattice_sigma_normalized_score(
            noisy_lattice_parameters, real_lattice_parameters, sigmas_n
        )
        return target_normalized_scores

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        output = self._generic_step(batch, batch_idx)
        batch_size = self._get_batch_size(batch)

        list_losses_to_log = [output["loss"]]
        list_labels = ["loss"]

        if "regularizer_loss" in output:
            list_losses_to_log.append(output["regularizer_loss"])
            list_labels.append("regularizer_loss")

        for loss, label in zip(list_losses_to_log, list_labels):
            # The 'train_step_loss' is only logged on_step, meaning it is a value for each batch
            self.log(f"train_step_{label}", loss, on_step=True, on_epoch=False, prog_bar=True)

            # The 'train_epoch_loss' is aggregated (batch_size weighted average) and logged once per epoch.
            self.log(
                f"train_epoch_{label}",
                loss,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

        for axl_field, axl_name in AXL_NAME_DICT.items():
            self.log(
                f"train_epoch_{axl_name}_loss",
                getattr(output["unreduced_loss"], axl_field).mean(),
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
        return output

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
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

        for axl_field, axl_name in AXL_NAME_DICT.items():
            self.log(
                f"validation_epoch_{axl_name}_loss",
                getattr(output["unreduced_loss"], axl_field).mean(),
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

        if not self.draw_samples:
            return output

        if self.draw_samples and self.metrics_parameters.compute_energies:
            reference_energies = batch["potential_energy"]
            self.energy_ks_metric.register_reference_samples(reference_energies.cpu())

        if self.draw_samples and self.metrics_parameters.compute_structure_factor:
            basis_vectors = torch.diag_embed(batch["box"])  # TODO replace with AXL L
            cartesian_positions = get_positions_from_coordinates(
                relative_coordinates=output[AXL_COMPOSITION].X,
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

        for axl_field, axl_name in AXL_NAME_DICT.items():
            self.log(
                f"test_epoch_{axl_name}_loss",
                getattr(output["unreduced_loss"], axl_field).mean(),
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

        return output

    def generate_samples(self):
        """Generate a batch of samples."""
        # TODO add atom types generation
        assert (
            self.hyper_params.diffusion_sampling_parameters is not None
        ), "sampling parameters must be provided to create a generator."
        with ((torch.no_grad())):
            logger.info("Creating Generator for sampling")

            trajectory_initializer = instantiate_trajectory_initializer(
                self.hyper_params.diffusion_sampling_parameters.sampling_parameters)

            self.generator = instantiate_generator(
                sampling_parameters=self.hyper_params.diffusion_sampling_parameters.sampling_parameters,
                noise_parameters=self.hyper_params.diffusion_sampling_parameters.noise_parameters,
                axl_network=self.axl_network,  # TODO use A and L too
                trajectory_initializer=trajectory_initializer
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
        logger.info("Ending validation.")
        if not self.draw_samples:
            return

        logger.info("   - Drawing samples at the end of the validation epoch.")
        samples_batch = self.generate_samples()  # TODO generate atom types too

        if self.draw_samples and self.metrics_parameters.compute_energies:
            logger.info("       * Computing sample energies")
            sample_energies, _ = self.oracle.compute_oracle_energies_and_forces(samples_batch)
            logger.info("       * Registering sample energies")
            self.energy_ks_metric.register_predicted_samples(sample_energies.cpu())

            logger.info("       * Computing KS distance for energies")
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
            logger.info("       * Done logging KS distance for energies")

        if self.draw_samples and self.metrics_parameters.compute_structure_factor:
            logger.info("       * Computing sample distances")
            sample_distances = compute_distances_in_batch(
                cartesian_positions=samples_batch[
                    CARTESIAN_POSITIONS
                ],  # TODO replace with AXL
                unit_cell=samples_batch[UNIT_CELL],
                max_distance=self.metrics_parameters.structure_factor_max_distance,
            )

            logger.info("       * Registering sample distances")
            self.structure_ks_metric.register_predicted_samples(sample_distances.cpu())

            logger.info("       * Computing KS distance for distances")
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
            logger.info("       * Done logging sample distances")

    def on_validation_start(self) -> None:
        """On validation start."""
        logger.info("Starting validation.")

        logger.info("   - Clearing generator and metrics on validation start.")
        # Clear out any dangling state.
        self.generator = None
        if self.draw_samples and self.metrics_parameters.compute_energies:
            self.energy_ks_metric.reset()

        if self.draw_samples and self.metrics_parameters.compute_structure_factor:
            self.structure_ks_metric.reset()

    def on_train_start(self) -> None:
        """On train start."""
        logger.info("Starting train.")
        logger.info("   - Clearing generator and metrics.")
        # Clear out any dangling state.
        self.generator = None
        if self.draw_samples and self.metrics_parameters.compute_energies:
            self.energy_ks_metric.reset()

        if self.draw_samples and self.metrics_parameters.compute_structure_factor:
            self.structure_ks_metric.reset()

    def on_train_epoch_start(self) -> None:
        """On train epoch start."""
        logger.info(f"Starting Training Epoch {self.current_epoch}.")

    def on_train_epoch_end(self) -> None:
        """On train epoch end."""
        logger.info(f"Ending Training Epoch {self.current_epoch}.")

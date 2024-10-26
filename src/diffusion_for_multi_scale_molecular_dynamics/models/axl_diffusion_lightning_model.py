import logging
from dataclasses import dataclass
from typing import Any, Optional

import pytorch_lightning as pl
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.instantiate_generator import (
    instantiate_generator,
)
from diffusion_for_multi_scale_molecular_dynamics.metrics.kolmogorov_smirnov_metrics import (
    KolmogorovSmirnovMetrics,
)
from diffusion_for_multi_scale_molecular_dynamics.models.loss import (
    LossParameters,
    create_loss_calculator,
)
from diffusion_for_multi_scale_molecular_dynamics.models.optimizer import (
    OptimizerParameters,
    load_optimizer,
)
from diffusion_for_multi_scale_molecular_dynamics.models.scheduler import (
    SchedulerParameters,
    load_scheduler_dictionary,
)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import (
    ScoreNetworkParameters,
)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network_factory import (
    create_score_network,
)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES,
    AXL,
    AXL_NAME_DICT,
    CARTESIAN_FORCES,
    CARTESIAN_POSITIONS,
    NOISE,
    NOISY_AXL,
    ORIGINAL_AXL,
    RELATIVE_COORDINATES,
    TIME,
    UNIT_CELL,
)
from diffusion_for_multi_scale_molecular_dynamics.oracle.energies import (
    compute_oracle_energies,
)
from diffusion_for_multi_scale_molecular_dynamics.samplers.noisy_sampler import (
    NoisyAtomTypesSampler,
    NoisyLatticeSampler,
    NoisyRelativeCoordinatesSampler,
)
from diffusion_for_multi_scale_molecular_dynamics.samplers.variance_sampler import (
    NoiseParameters,
    NoiseScheduler,
)
from diffusion_for_multi_scale_molecular_dynamics.samples.diffusion_sampling_parameters import (
    DiffusionSamplingParameters,
)
from diffusion_for_multi_scale_molecular_dynamics.samples.sampling import (
    create_batch_of_samples,
)
from diffusion_for_multi_scale_molecular_dynamics.score.wrapped_gaussian_score import (
    get_sigma_normalized_score,
)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates,
    map_relative_coordinates_to_unit_cell,
)
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import (
    class_index_to_onehot,
)
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import (
    compute_distances_in_batch,
)
from diffusion_for_multi_scale_molecular_dynamics.utils.tensor_utils import (
    broadcast_batch_matrix_tensor_to_all_dimensions,
    broadcast_batch_tensor_to_all_dimensions,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AXLDiffusionParameters:
    """AXL (atom, position, lattice) Diffusion parameters."""

    score_network_parameters: ScoreNetworkParameters
    loss_parameters: LossParameters
    optimizer_parameters: OptimizerParameters
    scheduler_parameters: Optional[SchedulerParameters] = None
    noise_parameters: NoiseParameters
    num_atom_types: int  # number of atom types - excluding the MASK class
    # convergence parameter for the Ewald-like sum of the perturbation kernel for coordinates.
    kmax_target_score: int = 4
    diffusion_sampling_parameters: Optional[DiffusionSamplingParameters] = None


class AXLDiffusionLightningModel(pl.LightningModule):
    """AXL Diffusion Lightning Model.

    This lightning model can train a score network predict the noise for relative coordinates, atom types and lattice
    vectors.
    """

    def __init__(self, hyper_params: AXLDiffusionParameters):
        """Init method.

        This initializes the class.
        """
        super().__init__()

        self.hyper_params = hyper_params
        self.save_hyperparameters(
            logger=False
        )  # It is not the responsibility of this class to log its parameters.

        # the score network is expected to produce an output as an AXL namedtuple:
        # atom: unnormalized estimate of p(a_0 | a_t)
        # positions: estimate of \sigma \nabla_{x_t} p_{t|0}(x_t | x_0)
        # lattices: TODO
        self.score_network = create_score_network(hyper_params.score_network_parameters)

        # loss is an AXL object with one loss for each element (atom type, coordinate, lattice)
        self.loss_calculator = create_loss_calculator(hyper_params.loss_parameters)

        # noisy samplers for atom types, coordinates and lattice vectors
        self.noisy_samplers = AXL(
            A=NoisyAtomTypesSampler(),
            X=NoisyRelativeCoordinatesSampler(),
            L=NoisyLatticeSampler(),
        )

        self.noise_scheduler = NoiseScheduler(
            hyper_params.noise_parameters,
            num_classes=hyper_params.num_atom_types + 1,  # add 1 for the MASK class
        )

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

        In this implementation, we choose :math:`\lambda(t_ = \sigma(t)^2` (a standard choice from the literature), such
        that the score network and the target scores that are used are actually "sigma normalized" versions, ie,
        pre-multiplied by sigma.

        For the atom type diffusion, the loss is defined as:

        .. math::
            L_a = E_{a_0 ~ p_data} [ \sum_{t=2}^T E_{at ~ p_{t|0]}
                [D_{KL}[q(a_{t-1} | a_t, a_0) || p_theta(a_t | a_{t-1} - \lambda_CE log p_\theta(a_0 | a_t)]
                - E_{a1 ~ p_{t=1| 0}} log p_\theta(a_0 | a_1) ]

        The loss that is computed is a Monte Carlo estimate of L, where we sample a mini-batch of relative coordinates
        configurations {x0} and atom types {a_0}; each of these configurations is noised with a random t value,
        with corresponding {sigma(t)}, {xt}, {beta(t)} and {a(t)}. Note the :math:`beta(t)` is used to compute the true
        posterior :math:``q(a_{t-1} | a_t, a_0)` and :math:`p_\theta(a_{t-1} | a_t)` in the atom type loss.

        Args:
            batch : a dictionary that should contain a data sample.
            batch_idx :  index of the batch
            no_conditional (optional): if True, do not use the conditional option of the forward. Used for validation.

        Returns:
            output_dictionary : contains the loss, the predictions and various other useful tensors.
        """
        # The RELATIVE_COORDINATES have dimensions [batch_size, number_of_atoms, spatial_dimension].
        assert (
            RELATIVE_COORDINATES in batch
        ), f"The field '{RELATIVE_COORDINATES}' is missing from the input."

        assert (
            ATOM_TYPES in batch
        ), f"The field '{ATOM_TYPES}' is missing from the input."

        x0 = batch[RELATIVE_COORDINATES]
        shape = x0.shape
        assert len(shape) == 3, (
            f"the shape of the RELATIVE_COORDINATES array should be [batch_size, number_of_atoms, spatial_dimensions]. "
            f"Got shape = {shape}."
        )

        a0 = batch[ATOM_TYPES]
        batch_size = self._get_batch_size(batch)
        atom_shape = a0.shape
        assert len(atom_shape) == (
            f"the shape of the ATOM_TYPES array should be [batch_size, number_of_atoms]. "
            f"Got shape = {atom_shape}"
        )

        lvec0 = batch[UNIT_CELL]
        # TODO assert on shape

        noise_sample = self.noise_scheduler.get_random_noise_sample(batch_size)

        # noise_sample.sigma and has dimension [batch_size]. Broadcast these values to be of shape
        # [batch_size, number_of_atoms, spatial_dimension] , which can be interpreted as
        # [batch_size, (configuration)]. All the sigma values must be the same for a given configuration.
        sigmas = broadcast_batch_tensor_to_all_dimensions(
            batch_values=noise_sample.sigma, final_shape=shape
        )
        # we can now get noisy coordinates
        xt = self.noisy_samplers.X.get_noisy_relative_coordinates_sample(x0, sigmas)

        # to get noisy atom types, we need to broadcast the transition matrix q_bar from size
        # [num_atom_types, num_atom_types] to [batch_size, number_of_atoms, num_atom_types, num_atom_types]. All the
        # q_bar matrices must be the same for a given configuration.
        q_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
            batch_values=noise_sample.q_matrix, final_shape=atom_shape
        )
        q_bar_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
            batch_values=noise_sample.q_bar_matrix, final_shape=atom_shape
        )

        q_bar_tm1_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
            batch_values=noise_sample.q_bar_tm1_matrix, final_shape=atom_shape
        )

        # we also need the atom types to be one-hot vector and not a class index
        a0_onehot = class_index_to_onehot(a0, self.hyper_params.num_atom_types + 1)

        at = self.noisy_samplers.A.get_noisy_atom_types_sample(
            a0_onehot, q_bar_matrices
        )
        at_onehot = class_index_to_onehot(at, self.hyper_params.num_atom_types + 1)

        # TODO do the same for the lattice vectors
        lvect = self.noisy_samplers.L.get_noisy_lattice_vectors(lvec0)

        noisy_sample = AXL(A=at, X=xt, L=lvec0)  # not one-hot

        original_sample = AXL(A=a0, X=x0, L=lvect)

        # Get the loss targets
        # Coordinates: The target is nabla log p_{t|0} (xt | x0): it is NOT the "score", but rather a "conditional"
        # (on x0) score.
        target_coordinates_normalized_conditional_scores = (
            self._get_coordinates_target_normalized_score(xt, x0, sigmas)
        )
        # for the atom types, the loss is constructed from the Q and barQ matrices

        # TODO get unit_cell from the noisy version and not a kwarg in batch (at least replace with namespace name)
        unit_cell = torch.diag_embed(
            batch["box"]
        )  # from (batch, spatial_dim) to (batch, spatial_dim, spatial_dim)

        forces = batch[CARTESIAN_FORCES]

        augmented_batch = {
            NOISY_AXL: noisy_sample,
            TIME: noise_sample.time.reshape(-1, 1),
            NOISE: noise_sample.sigma.reshape(-1, 1),
            UNIT_CELL: unit_cell,  # TODO remove and take from AXL instead
            CARTESIAN_FORCES: forces,
        }

        use_conditional = None if no_conditional is False else False
        model_predictions = self.score_network(
            augmented_batch, conditional=use_conditional
        )
        # this output is expected to be an AXL object
        # X score network output: an estimate of the sigma normalized score for the coordinates,
        # A score network output: an unnormalized estimate of p(a_0 | a_t) for the atom types
        # TODO something for the lattice

        unreduced_loss_coordinates = self.loss_calculator[
            RELATIVE_COORDINATES
        ].calculate_unreduced_loss(
            model_predictions.X,
            target_coordinates_normalized_conditional_scores,
            sigmas,
        )

        unreduced_loss_atom_types = self.loss_calculator[
            ATOM_TYPES
        ].calculate_unreduced_loss(
            predicted_unnormalized_probabilities=model_predictions.A,
            one_hot_real_atom_types=a0_onehot,
            one_hot_noisy_atom_types=at_onehot,
            time_indices=noise_sample.indices,
            q_matrices=q_matrices,
            q_bar_matrices=q_bar_matrices,
            q_bar_tm1_matrices=q_bar_tm1_matrices,
        )

        # TODO placeholder - returns zero
        unreduced_loss_lattice = self.loss_calculator.L.calculate_unreduced_loss(
            model_predictions.L
        )

        # TODO consider having weights in front of each component
        aggregated_loss = (
            unreduced_loss_coordinates
            + unreduced_loss_lattice
            + unreduced_loss_atom_types
        )

        loss = torch.mean(aggregated_loss)

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
            loss=loss,
            sigmas=sigmas,
            model_predictions=model_predictions_detached,
            target_coordinates_normalized_conditional_scores=target_coordinates_normalized_conditional_scores,
        )
        output[ORIGINAL_AXL] = original_sample
        output[NOISY_AXL] = NOISY_AXL
        output[TIME] = augmented_batch[TIME]
        output[UNIT_CELL] = augmented_batch[
            UNIT_CELL
        ]  # TODO remove and use AXL instead

        return output

    def _get_coordinates_target_normalized_score(
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

        for axl_field in output["unreduced_loss"]._fields:
            self.log(
                f"train_epoch_{AXL_NAME_DICT[axl_field]}_loss",
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

        for axl_field in output["unreduced_loss"]._fields:
            self.log(
                f"validation_epoch_{AXL_NAME_DICT[axl_field]}_loss",
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
                relative_coordinates=batch[ORIGINAL_AXL].X,
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

        for axl_field in output["unreduced_loss"]._fields:
            self.log(
                f"test_epoch_{AXL_NAME_DICT[axl_field]}_loss",
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
        logger.info("Ending validation.")
        if not self.draw_samples:
            return

        logger.info("   - Drawing samples at the end of the validation epoch.")
        samples_batch = self.generate_samples()  # TODO generate atom types too

        if self.draw_samples and self.metrics_parameters.compute_energies:
            logger.info("       * Computing sample energies")
            sample_energies = compute_oracle_energies(samples_batch)
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

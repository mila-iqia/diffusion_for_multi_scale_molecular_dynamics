from dataclasses import dataclass
from typing import AnyStr, Dict, Optional

import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.repulsive_force import \
    RepulsiveForceParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.repulsive_force_factory import \
    create_repulsive_force
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISY_AXL_COMPOSITION, TIME)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates, get_reciprocal_basis_vectors,
    get_relative_coordinates_from_cartesian_positions,
    map_lattice_parameters_to_unit_cell_vectors)


@dataclass(kw_only=True)
class ForceFieldAugmentedScoreNetworkParameters:
    """Force field parameters.

    Args:
        score_network: The ScoreNetwork on which this class adds a repulsive score.
        repulsive_force_parameters: A RepulsiveForceParameters to calculate cartesian_forces.
        force_activation_scale: Define the order of magnitude between a strong repulsion and weak repulsion.
        use_for_training: If the RepulsionScore should be used during training. False means only at inference time.
    """
    repulsive_force_parameters: RepulsiveForceParameters
    force_activation_scale: float = 100.0
    use_for_training: bool = False


class ForceFieldAugmentedScoreNetwork(torch.nn.Module):
    """Force Field-Augmented Score Network.

    This class wraps around an arbitrary score network in order to augment
    its output with an effective "force field". The intuition behind this is that
    atoms should never be very close to each other, but random numbers can lead
    to such proximity: a repulsive force field will encourage atoms to separate during
    diffusion.
    """

    def __init__(
        self, score_network: ScoreNetwork, force_field_parameters: ForceFieldAugmentedScoreNetworkParameters
    ):
        """Wrapper around ScoreNetwork that adds a contribution to the predicted score.

        You can then add it to AXLDiffusionLightningModel with model.use_force_field_augmented_score_network

        Args:
            score_network: The ScoreNetwork on which this class adds a repulsive score.
            force_field_parameters: Parameters of the force_field
        """
        super().__init__()

        self._score_network = score_network
        self.repulsive_force = create_repulsive_force(force_field_parameters.repulsive_force_parameters)
        self._force_activation_scale = force_field_parameters.force_activation_scale
        self._use_for_training = force_field_parameters.use_for_training

    def forward(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: Optional[bool] = None
    ) -> AXL:
        """Model forward.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional: if True, do a conditional forward, if False, do a unconditional forward. If None, choose
                randomly with probability conditional_prob

        Returns:
            computed_scores : the scores computed by the model.
        """
        raw_scores = self._score_network(batch, conditional)
        if self.training and not self._use_for_training:
            return raw_scores

        force_directions, force_importance = self.get_force_score_from_batch(batch)

        eps = 1e-1  # If forces from the model are really small, we still want ZBL to have some impact
        overflow_security = 1e+16  # torch.float32 max val is e38 meaning norm is limited to e19
        raw_norm = torch.clamp(raw_scores.X,
                               -overflow_security,
                               overflow_security).norm(dim=(1, 2)).clamp_min(eps)

        force_scores = force_directions * raw_norm[:, None, None]

        # Mix the two scores together, keeping he didn't give us a raid message but hi raw_scores unchanged
        force_importance = force_importance.clamp(max=1 - 1e-6)  # force_importance=1 would cause inf
        force_prefactor = force_importance / (1 - force_importance)

        updated_X_scores = raw_scores.X + force_prefactor[:, None, None] * force_scores
        updated_scores = AXL(A=raw_scores.A, X=updated_X_scores, L=raw_scores.L)

        return updated_scores

    def get_force_score_from_batch(
        self, batch: Dict[AnyStr, torch.Tensor]
    ) -> torch.Tensor:
        """Get relative coordinates repulsive score derived from the repulsive forces.

        The score is divided into two quantities. The normalized forces gives the direction of the score
        and the analytical fraction gives its magnitude.

        normalized_forces = F / |F|,
        where |F| is the norm over each configuration individually.

        analytical_fraction indicates how strong the correction should be as a fraction of the total score.
        It takes into account :
         1. The strength of the forces with respect to force_activation_scale.
         2. The discretization time, as atoms overlapping at t=T are expected, but catastrophic at T=0.
        The formula linear w.r.t time (for now) :
            analytical_fraction = discretization_time * g
            g = <|F|> / (<|F|> + self.force_activation_scale)

        With this expression, the repulsion should smoothly appears when atoms become close and
        the diffusion time is getting closer to 0.

        Args:
            batch : dictionary containing the data to be processed by the model.

        Returns:
            normalized_forces: normalized repulsive score [Batch_size, Natoms, 3] (norm=1 for each configuration)
            analytical_fraction: repulsion score correction weight [Batch_size]
        """
        composition_i = batch[NOISY_AXL_COMPOSITION]
        time = batch[TIME]

        epsilon = 1e-12  # So force doesn't diverge if every atom is farther than radial_cutoff
        basis_vectors = map_lattice_parameters_to_unit_cell_vectors(composition_i.L)
        cartesian_positions = get_positions_from_coordinates(composition_i.X, basis_vectors)
        reciprocal_basis_vectors = get_reciprocal_basis_vectors(basis_vectors)

        cartesian_forces = self.repulsive_force.get_cartesian_forces(
            composition_i.A,
            cartesian_positions,
            basis_vectors
        )
        relative_forces = get_relative_coordinates_from_cartesian_positions(
            cartesian_forces, reciprocal_basis_vectors
        )

        normalization_over_batch = relative_forces.norm(dim=[1, 2]).clamp_min(epsilon)  # [B]
        normalized_relative_forces = relative_forces / normalization_over_batch[:, None, None]  # [B,N,3]
        g = normalization_over_batch / (normalization_over_batch + self._force_activation_scale)  # [B]

        analytical_fraction = (1 - time[:, 0]) * g[:]  # We want a constant scaling wrt sigma_i, =0 at T and max at 0

        return normalized_relative_forces, analytical_fraction

from dataclasses import dataclass
from typing import AnyStr, Dict, Optional

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import (
    ScoreNetwork,
)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL,
    NOISY_AXL_COMPOSITION,
    UNIT_CELL,
)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates,
    get_reciprocal_basis_vectors,
    get_relative_coordinates_from_cartesian_positions,
)
from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import (
    AdjacencyInfo,
    get_periodic_adjacency_information,
)


@dataclass(kw_only=True)
class ForceFieldParameters:
    """Force field parameters.

    The force field is based on a potential of the form:

        phi(r) = strength * (r - radial_cutoff)^2

    The corresponding force is thus of the form
        F(r) = -nabla phi(r) = -2 strength * ( r - radial_cutoff) r_hat.
    """

    radial_cutoff: float  # Cutoff to the interaction, in Angstrom
    strength: float  # Strength of the repulsion


class ForceFieldAugmentedScoreNetwork(torch.nn.Module):
    """Force Field-Augmented Score Network.

    This class wraps around an arbitrary score network in order to augment
    its output with an effective "force field". The intuition behind this is that
    atoms should never be very close to each other, but random numbers can lead
    to such proximity: a repulsive force field will encourage atoms to separate during
    diffusion.
    """

    def __init__(
        self, score_network: ScoreNetwork, force_field_parameters: ForceFieldParameters
    ):
        """Init method.

        Args:
            score_network : a score network, to be augmented with a repulsive force.
            force_field_parameters : parameters for the repulsive force.
        """
        super().__init__()

        self._score_network = score_network
        self._force_field_parameters = force_field_parameters

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
        forces = self.get_relative_coordinates_pseudo_force(batch)
        updated_scores = AXL(A=raw_scores.A, X=raw_scores.X + forces, L=raw_scores.L)
        return updated_scores

    def _get_cartesian_pseudo_forces_contributions(
        self, cartesian_displacements: torch.Tensor
    ):
        """Get cartesian pseudo forces.

        The potential is given by
            phi(r) = s * (r - r0)^2

        Args:
            cartesian_displacements : vectors (r_i - r_j). Dimension [number_of_edges, spatial_dimension]

        Returns:
            cartesian_pseudo_forces_contributions: Force contributions for each displacement, for the
                chosen potential. F(r_i - r_j) = - d/dr phi(r) (r_i - r_j) / ||r_i - r_j||
        """
        s = self._force_field_parameters.strength
        r0 = self._force_field_parameters.radial_cutoff

        number_of_edges, spatial_dimension = cartesian_displacements.shape

        r = torch.linalg.norm(cartesian_displacements, dim=1)

        # Add a small epsilon value in case r is close to zero, to avoid NaNs.
        epsilon = torch.tensor(1.0e-8).to(r)

        pseudo_force_prefactors = 2.0 * s * (r - r0) / (r + epsilon)
        # Repeat so we can multiply by r_hat
        repeat_pseudo_force_prefactors = einops.repeat(
            pseudo_force_prefactors, "e -> e d", d=spatial_dimension
        )
        contributions = repeat_pseudo_force_prefactors * cartesian_displacements
        return contributions

    def _get_adjacency_information(
        self, batch: Dict[AnyStr, torch.Tensor]
    ) -> AdjacencyInfo:
        basis_vectors = batch[UNIT_CELL]
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        cartesian_positions = get_positions_from_coordinates(
            relative_coordinates, basis_vectors
        )

        adj_info = get_periodic_adjacency_information(
            cartesian_positions,
            basis_vectors,
            radial_cutoff=self._force_field_parameters.radial_cutoff,
        )
        return adj_info

    def _get_cartesian_displacements(
        self, adj_info: AdjacencyInfo, batch: Dict[AnyStr, torch.Tensor]
    ):
        # The following are 1D arrays of length equal to the total number of neighbors for all batch elements
        # and all atoms.
        #   bch: which batch does an edge belong to
        #   src: at which atom does an edge start
        #   dst: at which atom does an edge end
        bch = adj_info.edge_batch_indices
        src, dst = adj_info.adjacency_matrix

        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        basis_vectors = batch[UNIT_CELL]  # TODO replace with AXL L
        cartesian_positions = get_positions_from_coordinates(
            relative_coordinates, basis_vectors
        )

        cartesian_displacements = (
            cartesian_positions[bch, dst]
            - cartesian_positions[bch, src]
            + adj_info.shifts
        )
        return cartesian_displacements

    def _get_cartesian_pseudo_forces(
        self,
        cartesian_pseudo_force_contributions: torch.Tensor,
        adj_info: AdjacencyInfo,
        batch: Dict[AnyStr, torch.Tensor],
    ):
        # The following are 1D arrays of length equal to the total number of neighbors for all batch elements
        # and all atoms.
        #   bch: which batch does an edge belong to
        #   src: at which atom does an edge start
        #   dst: at which atom does an edge end
        bch = adj_info.edge_batch_indices
        src, dst = adj_info.adjacency_matrix

        batch_size, natoms, spatial_dimension = batch[NOISY_AXL_COMPOSITION].X.shape

        # Combine the bch and src index into a single global index
        node_idx = natoms * bch + src

        list_pseudo_force_components = []

        for space_idx in range(spatial_dimension):
            pseudo_force_component = torch.zeros(natoms * batch_size).to(
                cartesian_pseudo_force_contributions
            )
            pseudo_force_component.scatter_add_(
                dim=0,
                index=node_idx,
                src=cartesian_pseudo_force_contributions[:, space_idx],
            )
            list_pseudo_force_components.append(pseudo_force_component)

        cartesian_pseudo_forces = einops.rearrange(
            list_pseudo_force_components,
            pattern="d (b n) -> b n d",
            b=batch_size,
            n=natoms,
        )
        return cartesian_pseudo_forces

    def get_relative_coordinates_pseudo_force(
        self, batch: Dict[AnyStr, torch.Tensor]
    ) -> torch.Tensor:
        """Get relative coordinates pseudo force.

        Args:
            batch : dictionary containing the data to be processed by the model.

        Returns:
            relative_pseudo_forces : repulsive force in relative coordinates.
        """
        adj_info = self._get_adjacency_information(batch)

        cartesian_displacements = self._get_cartesian_displacements(adj_info, batch)
        cartesian_pseudo_force_contributions = (
            self._get_cartesian_pseudo_forces_contributions(cartesian_displacements)
        )

        cartesian_pseudo_forces = self._get_cartesian_pseudo_forces(
            cartesian_pseudo_force_contributions, adj_info, batch
        )

        basis_vectors = batch[UNIT_CELL]  # TODO replace with AXL L
        reciprocal_basis_vectors = get_reciprocal_basis_vectors(basis_vectors)
        relative_pseudo_forces = get_relative_coordinates_from_cartesian_positions(
            cartesian_pseudo_forces, reciprocal_basis_vectors
        )

        return relative_pseudo_forces

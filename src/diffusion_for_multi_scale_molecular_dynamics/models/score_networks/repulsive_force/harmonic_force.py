from __future__ import annotations

from dataclasses import dataclass

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.repulsive_force import (
    RepulsiveForce, RepulsiveForceParameters)
from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import (
    AdjacencyInfo, get_periodic_adjacency_information)


@dataclass(kw_only=True)
class HarmonicForceParameters(RepulsiveForceParameters):
    """Specific Hyper-parameters for the harmonic force field."""
    architecture: str = "harmonic"
    strength: float  # Strength of the repulsion


class HarmonicForce(RepulsiveForce):
    """Harmonic potential to get an analytical repulsion score."""

    def __init__(self, hyper_params: HarmonicForceParameters):
        """Initialize the analytical repulsion model which calculates forces and gives an analytical score."""
        self._force_field_strength = hyper_params.strength
        super().__init__(hyper_params)

    def get_cartesian_forces(self, A, cartesian_positions, basis_vectors) -> torch.Tensor:
        """Get cartesian coordinates pseudo force.

        Args:
            batch : dictionary containing the data to be processed by the model.

        Returns:
            cartesian_pseudo_forces : repulsive force in cartesian coordinates.
        """
        adj_info = get_periodic_adjacency_information(
            cartesian_positions,
            basis_vectors,
            radial_cutoff=self.radial_cutoff,
        )

        cartesian_displacements = self._get_cartesian_displacements(adj_info, cartesian_positions, basis_vectors)
        cartesian_pseudo_force_contributions = (
            self._get_cartesian_pseudo_forces_contributions(cartesian_displacements)
        )

        cartesian_pseudo_forces = self._get_cartesian_pseudo_forces(
            cartesian_pseudo_force_contributions, adj_info, cartesian_positions
        )

        return cartesian_pseudo_forces

    def _get_cartesian_displacements(
        self, adj_info: AdjacencyInfo, cartesian_positions, basis_vectors
    ):
        # The following are 1D arrays of length equal to the total number of neighbors for all batch elements
        # and all atoms.
        #   bch: which batch does an edge belong to
        #   src: at which atom does an edge start
        #   dst: at which atom does an edge end
        bch = adj_info.edge_batch_indices
        src, dst = adj_info.adjacency_matrix

        cartesian_displacements = (
            cartesian_positions[bch, dst]
            - cartesian_positions[bch, src]
            + adj_info.shifts
        )
        return cartesian_displacements

    def _get_cartesian_pseudo_forces_contributions(self, cartesian_displacements):
        """Get cartesian pseudo forces.

        The force field is based on a potential of the form:
            phi(r) = strength * (r - radial_cutoff)^2

        The corresponding force is thus of the form
            F(r) = -nabla phi(r) = -2 strength * (r - radial_cutoff) r_hat.

        Args:
            cartesian_displacements : vectors (r_i - r_j). Dimension [number_of_edges, spatial_dimension]

        Returns:
            cartesian_pseudo_forces_contributions: Force contributions for each displacement, for the
                chosen potential. F(r_i - r_j) = - d/dr phi(r) (r_i - r_j) / ||r_i - r_j||
        """
        s = self._force_field_strength
        r0 = self.radial_cutoff

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

    def _get_cartesian_pseudo_forces(
        self,
        cartesian_pseudo_force_contributions: torch.Tensor,
        adj_info: AdjacencyInfo,
        cartesian_positions,
    ):
        # The following are 1D arrays of length equal to the total number of neighbors for all batch elements
        # and all atoms.
        #   bch: which batch does an edge belong to
        #   src: at which atom does an edge start
        #   dst: at which atom does an edge end
        bch = adj_info.edge_batch_indices
        src, dst = adj_info.adjacency_matrix

        batch_size, natoms, spatial_dimension = cartesian_positions.shape

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

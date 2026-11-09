from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import (
    AdjacencyInfo, get_periodic_adjacency_information)


@dataclass(kw_only=True)
class RepulsiveForceParameters:
    """Hyper-parameters for repulsive forces.

    Args:
        radial_cutoff (ang): The minimal interatomic distance with no contribution from this analytical model.
    """
    radial_cutoff: float


class RepulsiveForce(ABC, torch.nn.Module):
    """Analytical Atomic Repulsion Score.

    This class calculates a score based on an analytical model to helps stability.
    """

    def __init__(self, hyper_params: RepulsiveForceParameters):
        """Init method."""
        super().__init__()
        self.radial_cutoff = hyper_params.radial_cutoff

    def get_atomic_distances(
        self, cartesian_positions: torch.Tensor, basis_vectors: torch.Tensor
    ) -> tuple[AdjacencyInfo, torch.Tensor]:
        """Return the atomic distances between every neighbor pair within the radial cutoff.

        Args:
            cartesian_positions: Cartesian positions [nconf, natoms, 3]
            basis_vectors: Cell basis vectors [nconf, 3, 3]

        Returns:
            adj: AdjacencyInfo with source/destination indices, batch indices, etc.
            distances: interatomic distances for each edge [number_of_edges]
        """
        adj = get_periodic_adjacency_information(
            cartesian_positions=cartesian_positions,
            basis_vectors=basis_vectors,
            radial_cutoff=self.radial_cutoff,
            spatial_dimension=3,
        )
        b = adj.edge_batch_indices
        src, dst = adj.adjacency_matrix
        # Recompute distances from positions explicitly so autograd can differentiate through them.
        # adj.squared_distances comes from KeOps Kmin_argKmin does not support backward which is needed for autograd
        distances = torch.linalg.norm(
            cartesian_positions[b, dst] - cartesian_positions[b, src] + adj.shifts, dim=-1
        )
        return adj, distances

    @abstractmethod
    def get_cartesian_forces(self, A, cartesian_positions, basis_vectors):
        """Returns the cartesian forces."""
        pass

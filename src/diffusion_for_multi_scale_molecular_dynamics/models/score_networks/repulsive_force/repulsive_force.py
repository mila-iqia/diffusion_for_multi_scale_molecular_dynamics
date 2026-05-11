from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import \
    get_periodic_adjacency_information


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

    def get_atomic_distances(self, cartesian_positions, basis_vectors):
        """Return the atomic distance between every pair of atoms up to radial_cutoff. Else, gives -1.

        Args:
            cartesian_positions: Cartesian positions [nconf, natoms, 3]
            basis_vectors: Cell basis vectors [nconf, 3, 3]

        Returns:
            atomic_distances: [nconf, natoms, natoms]
                - d_ij if atom j is within cutoff of atom i (with PBC)
                - -1 otherwise
        """
        nconf, natoms, _ = cartesian_positions.shape

        atomic_distances = torch.full(
            (nconf, natoms, natoms),
            -1.0,
            requires_grad=False,
            dtype=cartesian_positions.dtype,
            device=cartesian_positions.device,
        )
        adj = get_periodic_adjacency_information(
            cartesian_positions=cartesian_positions,
            basis_vectors=basis_vectors,
            radial_cutoff=self.radial_cutoff,
            spatial_dimension=3,
        )

        src = adj.adjacency_matrix[0]
        dst = adj.adjacency_matrix[1]
        b = adj.edge_batch_indices  # b indicates the index of the configuration
        shift = adj.shifts
        xang1 = cartesian_positions[b, src]
        xang2 = cartesian_positions[b, dst] + shift
        distance_unordered = torch.linalg.norm(xang1 - xang2, dim=-1)

        atomic_distances = atomic_distances.index_put((b, src, dst), distance_unordered)
        atomic_distances = atomic_distances.index_put((b, dst, src), distance_unordered)

        return atomic_distances

    @abstractmethod
    def get_cartesian_forces(self, A, cartesian_positions, basis_vectors):
        """Returns the cartesian forces."""
        pass

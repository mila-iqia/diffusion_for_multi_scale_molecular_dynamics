"""EGNN on cartesian positions, with periodic edges built from the positions."""

from typing import Optional

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn import (
    E_GCL, EGNN)
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn_utils import \
    build_graph_periodic_edges_batch


class CartesianE_GCL(E_GCL):
    """E_GCL whose messages also see the direction of the bond between nodes i and j, relative to the cell axes."""

    number_of_edge_features = 2
    bond_direction_length = 0.1  # Angstrom: smooths the bond direction of nearly overlapping atoms, where it is
    # ill-defined and can swing abruptly from one orientation to another.

    def get_edge_features(self, radial: torch.Tensor, coord_diff: torch.Tensor) -> torch.Tensor:
        """Get the edge features of the messages: the distance squared and the bond direction.

        The bond direction is the cubic invariant (dx^4 + dy^4 + dz^4) / (d^2 + a^2)^2 of x_i - x_j = (dx, dy, dz), with
        a = bond_direction_length. For a bond much longer than a, it is 1 along <100> and 1/3 along <111>; it goes
        smoothly to zero when the atoms overlap.

        Args:
            radial: distance squared between nodes i and j. size: number of edges, 1
            coord_diff: difference between coordinates, x_i - x_j, before any normalization.
                size: number of edges, spatial dimension

        Returns:
            edge_features: the distance squared and the bond direction of every edge. size: number of edges, 2
        """
        bond_direction = (coord_diff**4).sum(dim=1, keepdim=True) / (radial + self.bond_direction_length**2) ** 2
        return torch.cat([radial, bond_direction], dim=1)


class CartesianEGNN(EGNN):
    """EGNN on cartesian positions, with periodic edges built from the positions."""

    def __init__(self, rebuild_edges_every_layer: bool = True, bond_direction_feature: bool = True, **egnn_arguments):
        """Init method.

        Args:
            rebuild_edges_every_layer: if True, rebuild the periodic edges from the layer-updated positions before
                every layer. If False, build them once from the input positions. Defaults to True.
            bond_direction_feature: if True, the messages also see the bond direction (see CartesianE_GCL).
                Defaults to True.
            egnn_arguments: the arguments of EGNN.
        """
        super().__init__(graph_layer_class=CartesianE_GCL if bond_direction_feature else E_GCL, **egnn_arguments)
        self.rebuild_edges_every_layer = rebuild_edges_every_layer
        self.bond_direction_feature = bond_direction_feature

    def _build_edges(
        self,
        x: torch.Tensor,
        cell_lengths: torch.Tensor,
        natoms: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build the periodic edges within radial_cutoff.

        Args:
            x: node coordinates. size is number of nodes, spatial dimension
            cell_lengths: per-sample orthogonal cell side lengths. size: batch_size, spatial_dimension
            natoms: optional real (non-padded) atom count per sample. size: batch_size

        Returns:
            edges as [i, j, shift_x, shift_y, shift_z]. size: number of edges, 2 + spatial_dimension
        """
        batch_size = cell_lengths.shape[0]
        batched_x = einops.rearrange(
            x, "(batch natoms) spatial_dimension -> batch natoms spatial_dimension",
            batch=batch_size, natoms=x.shape[0] // batch_size,
        )
        return build_graph_periodic_edges_batch(
            batched_x, cell_lengths, self.radial_cutoff, natoms=natoms,
        )

    @staticmethod
    def _get_edge_distances_squared(edges: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Get the distance squared of every periodic edge.

        Args:
            edges: edges as [i, j, shift_x, shift_y, shift_z]. size: number of edges, 2 + spatial_dimension
            x: node coordinates. size is number of nodes, spatial dimension

        Returns:
            distance squared of every edge. size: number of edges, 1
        """
        row, col = edges[:, 0].long(), edges[:, 1].long()
        return ((x[row] - x[col] - edges[:, 2:]) ** 2).sum(dim=1, keepdim=True)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        cell_lengths: torch.Tensor,
        natoms: Optional[torch.Tensor] = None,
    ) -> AXL:
        """Forward instructions for the model.

        Args:
            h: node features. size is number of nodes (atoms), input size
            x: cartesian node coordinates. size is number of nodes, spatial dimension
            cell_lengths: per-sample orthogonal cell side lengths, used to build the periodic
                edges. size: batch_size, spatial_dimension
            natoms: optional real (non-padded) atom count per sample. size: batch_size

        Returns:
            estimated score in an AXL namedtuple (see EGNN._get_model_outputs).
        """
        h = self.embedding_in(h)
        edges = self._build_edges(x, cell_lengths, natoms)
        for layer_index, graph_layer in enumerate(self.graph_layers):
            if self.rebuild_edges_every_layer and layer_index > 0:
                edges = self._build_edges(x, cell_lengths, natoms)
            h, x = graph_layer(h, edges, x, self._get_edge_distances_squared(edges, x))
        return self._get_model_outputs(h, x)

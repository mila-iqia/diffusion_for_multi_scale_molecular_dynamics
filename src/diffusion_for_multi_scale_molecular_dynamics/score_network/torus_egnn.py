"""EGNN on positions embedded on the torus, with fixed edges given with their cartesian distances."""

import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn import \
    EGNN


class TorusEGNN(EGNN):
    """EGNN on positions embedded on the torus, with fixed edges given with their cartesian distances."""

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edges: torch.Tensor,
        edge_distances_squared: torch.Tensor,
    ) -> AXL:
        """Forward instructions for the model.

        Args:
            h: node features. size is number of nodes (atoms), input size
            x: node coordinates embedded on the torus. size is number of nodes, embedding dimension
            edges: edges as [i, j], used by every layer. size: number of edges, 2
            edge_distances_squared: cartesian distance squared of every edge (Angstrom^2), for the smooth cutoff.
                size: number of edges, 1

        Returns:
            estimated score in an AXL namedtuple (see EGNN._get_model_outputs).
        """
        h = self.embedding_in(h)
        for graph_layer in self.graph_layers:
            h, x = graph_layer(h, edges, x, edge_distances_squared)
        return self._get_model_outputs(h, x)

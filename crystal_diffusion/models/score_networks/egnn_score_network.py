from dataclasses import dataclass
from typing import AnyStr, Dict

import einops
import torch

from crystal_diffusion.models.egnn_clean import EGNN, get_edges_batch
from crystal_diffusion.models.score_networks import ScoreNetworkParameters
from crystal_diffusion.models.score_networks.score_network import ScoreNetwork
from crystal_diffusion.namespace import NOISE, NOISY_RELATIVE_COORDINATES


@dataclass(kw_only=True)
class EGNNScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for ENN score networks."""
    architecture: str = "egnn"
    normalize: bool = False
    hidden_dimensions_size: int
    number_of_layers: int


class EGNNScoreNetwork(ScoreNetwork):
    """Score network using EGNN.

    This implementation leverages the baseline EGNN architecture from Satorras et al.
    It assumes that all "graphs" have the same number of nodes and are fully connected.
    """

    def __init__(self, hyper_params: EGNNScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(EGNNScoreNetwork, self).__init__(hyper_params)

        self.number_of_features_per_node = 1
        self.spatial_dimension = hyper_params.spatial_dimension

        self.projection_matrices = self._create_block_diagonal_projection_matrices(self.spatial_dimension)

        self.egnn = EGNN(
            in_node_nf=self.number_of_features_per_node,
            hidden_nf=hyper_params.hidden_dimensions_size,
            n_layers=hyper_params.number_of_layers,
            out_node_nf=1,
            in_edge_nf=1,
            normalize=hyper_params.normalize,
        )

    @staticmethod
    def _create_block_diagonal_projection_matrices(spatial_dimension: int) -> torch.Tensor:
        """Create block diagonal projection matrices.

        This method creates the "Gamma" matrices that are needed to project the higher dimensional
        outputs of EGNN back to the relevant dimension for the normalized score.

        The normalized score is defined as
            S = z . Gamma^alpha . hat_z
        where hat_z is the output of the EGNN model, z are the uplifted Euclidean position. This method
        creates the Gamma matrices, expressed as a tensor Gamma_{alpha, i, j}, where "alpha" is a coordinate
        in real space, and i,j are coordinates in the uplifted Euclidean space.

        Args:
            spatial_dimension : spatial dimension of the crystal.

        Returns:
            projection_matrices: block diagonal projection matrices, of
                dimension [spatial_dimension, 2 x spatial_dimension, 2 x spatial_dimension].
        """
        zeros = torch.zeros(2, 2)
        dimensional_projector = torch.tensor([[0., -1.], [1., 0.]])

        projection_matrices = []
        for space_idx in range(spatial_dimension):
            blocks = space_idx * [zeros] + [dimensional_projector] + (spatial_dimension - space_idx - 1) * [zeros]
            projection_matrices.append(torch.block_diag(*blocks))

        return torch.stack(projection_matrices)

    @staticmethod
    def _get_node_attributes(batch: Dict[AnyStr, torch.Tensor]) -> torch.Tensor:
        """Get node attributes.

        This method extracts the node atttributes, "h", to be fed as input to the EGNN network.
        Args:
            batch : the batch dictionary

        Returns:
            node_attributes: a tensor of dimension [number_of_nodes, number_for_features_per_node]
        """
        relative_coordinates = batch[NOISY_RELATIVE_COORDINATES]
        batch_size, number_of_atoms, spatial_dimension = relative_coordinates.shape

        sigmas = batch[NOISE].to(relative_coordinates.device)
        repeated_sigmas = einops.repeat(
            sigmas, "batch 1 -> (batch natoms) 1", natoms=number_of_atoms
        )
        return repeated_sigmas

    @staticmethod
    def _get_euclidean_positions(flat_relative_coordinates: torch.Tensor) -> torch.Tensor:
        """Get Euclidean positions.

        Get the positions that take points on the torus into a higher dimensional
        (ie, 2 x spatial_dimension) Euclidean space.

        Args:
            flat_relative_coordinates: relative coordinates, of dimensions [number_of_nodes, spatial_dimension]

        Returns:
            euclidean_positions : uplifted relative coordinates to a higher dimensional Euclidean space.
            euclidean_projectors : projectors in the higher dimensional Euclidean space to extract normalized score.
        """
        # Uplift the relative coordinates to the embedding Euclidean space
        angles = 2.0 * torch.pi * flat_relative_coordinates
        cosines = angles.cos()
        sines = angles.sin()
        euclidean_positions = einops.rearrange(
            torch.stack([cosines, sines]),
            "two nodes spatial_dimension -> nodes (spatial_dimension two)",
        )
        return euclidean_positions

    def _forward_unchecked(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> torch.Tensor:
        relative_coordinates = batch[NOISY_RELATIVE_COORDINATES]
        batch_size, number_of_atoms, spatial_dimension = relative_coordinates.shape

        edges, edge_attr = get_edges_batch(
            n_nodes=number_of_atoms, batch_size=batch_size
        )

        edges = [edge.to(relative_coordinates.device) for edge in edges]
        edge_attr = edge_attr.to(relative_coordinates.device)

        flat_relative_coordinates = einops.rearrange(
            relative_coordinates, "batch natom spatial_dimension -> (batch natom) spatial_dimension"
        )

        # Uplift the relative coordinates to the embedding Euclidean space.
        #   Dimensions [number_of_nodes, 2 x spatial_dimension]
        euclidean_positions = self._get_euclidean_positions(flat_relative_coordinates)

        node_attributes_h = self._get_node_attributes(batch)
        # The raw normalized score has dimensions [number_of_nodes, 2 x spatial_dimension]
        # CAREFUL! It is important to pass a clone of the euclidian positions because EGNN will modify its input!
        _, raw_normalized_score = self.egnn(
            h=node_attributes_h, x=euclidean_positions.clone(), edges=edges, edge_attr=edge_attr
        )

        # The projected score is defined a
        #       S^alpha = z . Gamma^alpha . hat_z
        #  where:
        #       - alpha is a spatial index (ie, x, y z) in the real space
        #       - z is  the uplifted "positions" in the 2 x spatial_dimension Euclidean space
        #       - hat_z is the output of the EGNN model, also in 2 x spatial_dimension
        #       - Gamma^alpha are the projection matrices
        flat_normalized_scores = einops.einsum(euclidean_positions, self.projection_matrices, raw_normalized_score,
                                               "nodes i, alpha i j, nodes j-> nodes alpha")

        normalized_scores = einops.rearrange(
            flat_normalized_scores,
            "(batch natoms) spatial_dimension -> batch natoms spatial_dimension",
            batch=batch_size,
            natoms=number_of_atoms
        )
        return normalized_scores

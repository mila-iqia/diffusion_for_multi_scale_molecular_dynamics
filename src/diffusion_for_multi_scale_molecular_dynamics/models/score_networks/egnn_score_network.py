from dataclasses import dataclass
from typing import AnyStr, Dict, Union

import einops
import torch
from crystal_diffusion.models.egnn import EGNN
from crystal_diffusion.models.egnn_utils import (get_edges_batch,
                                                 get_edges_with_radial_cutoff)
from crystal_diffusion.models.score_networks import ScoreNetworkParameters
from crystal_diffusion.models.score_networks.score_network import ScoreNetwork
from crystal_diffusion.namespace import (NOISE, NOISY_RELATIVE_COORDINATES,
                                         UNIT_CELL)


@dataclass(kw_only=True)
class EGNNScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for ENN score networks."""
    architecture: str = "egnn"
    message_n_hidden_dimensions: int = 1
    message_hidden_dimensions_size: int = 16
    node_n_hidden_dimensions: int = 1
    node_hidden_dimensions_size: int = 32
    coordinate_n_hidden_dimensions: int = 1
    coordinate_hidden_dimensions_size: int = 32
    residual: bool = True
    attention: bool = False
    normalize: bool = False
    tanh: bool = False
    coords_agg: str = "mean"
    message_agg: str = "mean"
    n_layers: int = 4
    edges: str = 'fully_connected'
    radial_cutoff: Union[float, None] = None
    drop_duplicate_edges: bool = True


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

        projection_matrices = self._create_block_diagonal_projection_matrices(self.spatial_dimension)
        self.register_parameter('projection_matrices',
                                torch.nn.Parameter(projection_matrices, requires_grad=False))

        self.edges = hyper_params.edges
        assert self.edges in ["fully_connected", "radial_cutoff"], \
            f'Edges type should be fully_connected or radial_cutoff. Got {self.edges}'

        self.radial_cutoff = hyper_params.radial_cutoff

        if self.edges == "fully_connected":
            assert self.radial_cutoff is None, "Specifying a radial cutoff is inconsistent with edges=fully_connected."
        else:
            assert type(self.radial_cutoff) is float, \
                "A floating point value for the radial cutoff is needed for edges=radial_cutoff."

        self.drop_duplicate_edges = hyper_params.drop_duplicate_edges

        self.egnn = EGNN(
            input_size=self.number_of_features_per_node,
            message_n_hidden_dimensions=hyper_params.message_n_hidden_dimensions,
            message_hidden_dimensions_size=hyper_params.message_hidden_dimensions_size,
            node_n_hidden_dimensions=hyper_params.node_n_hidden_dimensions,
            node_hidden_dimensions_size=hyper_params.node_hidden_dimensions_size,
            coordinate_n_hidden_dimensions=hyper_params.coordinate_n_hidden_dimensions,
            coordinate_hidden_dimensions_size=hyper_params.coordinate_hidden_dimensions_size,
            residual=hyper_params.residual,
            attention=hyper_params.attention,
            normalize=hyper_params.normalize,
            tanh=hyper_params.tanh,
            coords_agg=hyper_params.coords_agg,
            message_agg=hyper_params.message_agg,
            n_layers=hyper_params.n_layers,
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

        if self.edges == "fully_connected":
            edges = get_edges_batch(
                n_nodes=number_of_atoms, batch_size=batch_size
            )
        else:
            edges = get_edges_with_radial_cutoff(
                relative_coordinates, batch[UNIT_CELL], self.radial_cutoff,
                drop_duplicate_edges=self.drop_duplicate_edges
            )

        edges = edges.to(relative_coordinates.device)

        flat_relative_coordinates = einops.rearrange(
            relative_coordinates, "batch natom spatial_dimension -> (batch natom) spatial_dimension"
        )

        # Uplift the relative coordinates to the embedding Euclidean space.
        #   Dimensions [number_of_nodes, 2 x spatial_dimension]
        euclidean_positions = self._get_euclidean_positions(flat_relative_coordinates)

        node_attributes_h = self._get_node_attributes(batch)
        # The raw normalized score has dimensions [number_of_nodes, 2 x spatial_dimension]
        # CAREFUL! It is important to pass a clone of the euclidian positions because EGNN will modify its input!
        raw_normalized_score = self.egnn(
            h=node_attributes_h, edges=edges, x=euclidean_positions.clone()
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

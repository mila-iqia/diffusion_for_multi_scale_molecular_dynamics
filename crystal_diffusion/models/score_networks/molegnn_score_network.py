from dataclasses import dataclass
from typing import AnyStr, Dict

import einops
import torch

from crystal_diffusion.models.egnn import EGNN
from crystal_diffusion.models.egnn_utils import get_edges_batch
from crystal_diffusion.models.score_networks import ScoreNetworkParameters
from crystal_diffusion.models.score_networks.score_network import ScoreNetwork
from crystal_diffusion.namespace import NOISE, NOISY_RELATIVE_COORDINATES


@dataclass(kw_only=True)
class MolEGNNScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for EGNN score networks for molecules."""
    architecture: str = "molegnn"
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
    n_layers: int = 4


class MolEGNNScoreNetwork(ScoreNetwork):
    """Score network using EGNN considering crystalline Si inputs as molecules (no periodicity).

    This implementation leverages the baseline EGNN architecture from Satorras et al.
    It assumes that all "graphs" have the same number of nodes and are fully connected.
    """

    def __init__(self, hyper_params: MolEGNNScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(MolEGNNScoreNetwork, self).__init__(hyper_params)

        self.number_of_features_per_node = 1
        self.spatial_dimension = hyper_params.spatial_dimension

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
            n_layers=hyper_params.n_layers,
        )

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

    def _forward_unchecked(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> torch.Tensor:
        relative_coordinates = batch[NOISY_RELATIVE_COORDINATES]
        batch_size, number_of_atoms, spatial_dimension = relative_coordinates.shape

        edges = get_edges_batch(
            n_nodes=number_of_atoms, batch_size=batch_size
        )

        edges = edges.to(relative_coordinates.device)

        flat_relative_coordinates = einops.rearrange(
            relative_coordinates, "batch natom spatial_dimension -> (batch natom) spatial_dimension"
        )

        node_attributes_h = self._get_node_attributes(batch)
        # The raw normalized score has dimensions [number_of_nodes, 2 x spatial_dimension]
        # CAREFUL! It is important to pass a clone of the euclidian positions because EGNN will modify its input!
        raw_normalized_score = self.egnn(
            h=node_attributes_h, edges=edges, x=flat_relative_coordinates.clone()
        )

        normalized_scores = einops.rearrange(
            raw_normalized_score,
            "(batch natoms) spatial_dimension -> batch natoms spatial_dimension",
            batch=batch_size,
            natoms=number_of_atoms
        )
        return normalized_scores

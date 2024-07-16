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
    architecture: str = 'egnn'
    hidden_dim: int


class EGNNScoreNetwork(ScoreNetwork):
    """Score network using EGNN."""

    def __init__(self, hyper_params: EGNNScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(EGNNScoreNetwork, self).__init__(hyper_params)

        self.egnn = EGNN(in_node_nf=1, hidden_nf=hyper_params.hidden_dim, out_node_nf=1, in_edge_nf=1)

    def _forward_unchecked(self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False) -> torch.Tensor:

        relative_coordinates = batch[NOISY_RELATIVE_COORDINATES]
        batch_size, number_of_atoms, spatial_dimension = relative_coordinates.shape

        edges, edge_attr = get_edges_batch(n_nodes=number_of_atoms, batch_size=batch_size)

        flat_relative_coordinates = einops.rearrange(relative_coordinates,
                                                     "batch natom space -> (batch natom) space")

        # Uplift the relative coordinates to the embedding Euclidean space
        angles = 2. * torch.pi * flat_relative_coordinates
        cosines = angles.cos()
        sines = angles.sin()
        z = einops.rearrange([cosines, sines], "type batch space -> batch (space type)")

        sigmas = batch[NOISE]
        repeated_sigmas = einops.repeat(sigmas, "batch 1 -> (batch natoms) 1", natoms=number_of_atoms)

        _, raw_normalized_score = self.egnn(h=repeated_sigmas, x=z, edges=edges, edge_attr=edge_attr)

        normalized_scores_1 = sines[:, 0] * raw_normalized_score[:, 0] - cosines[:, 0] * raw_normalized_score[:, 1]
        normalized_scores_2 = sines[:, 1] * raw_normalized_score[:, 2] - cosines[:, 1] * raw_normalized_score[:, 3]
        normalized_scores_3 = sines[:, 2] * raw_normalized_score[:, 4] - cosines[:, 2] * raw_normalized_score[:, 5]

        normalized_scores = einops.rearrange([normalized_scores_1, normalized_scores_2, normalized_scores_3],
                                             "space (batch natoms) -> batch natoms space",
                                             batch=batch_size, natoms=number_of_atoms)
        return normalized_scores

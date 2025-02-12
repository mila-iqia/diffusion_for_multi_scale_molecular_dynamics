from dataclasses import dataclass
from typing import AnyStr, Dict, Union

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.egnn import EGNN
from diffusion_for_multi_scale_molecular_dynamics.models.egnn_utils import (
    get_edges_batch, get_edges_with_radial_cutoff)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISE, NOISY_AXL_COMPOSITION, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    class_index_to_onehot
from diffusion_for_multi_scale_molecular_dynamics.models.gemnet.gemnet import GemNetT


@dataclass(kw_only=True)
class GemNetScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for ENN score networks."""

    architecture: str = "gemnet"
    latent_dim: int = 16
    atom_type_embedding_size: int = 16


class GemNetScoreNetwork(ScoreNetwork):
    """Score network using GemNet as the backbone.

    This implementation leverages the baseline EGNN architecture from Satorras et al.
    It assumes that all "graphs" have the same number of nodes and are fully connected.
    """

    def __init__(self, hyper_params: GemNetScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(GemNetScoreNetwork, self).__init__(hyper_params)

        self.spatial_dimension = hyper_params.spatial_dimension
        self.num_atom_types = hyper_params.num_atom_types
        self.num_classes = self.num_atom_types + 1
        self.number_of_features_per_node = (
            self.num_atom_types + 2
        )  # +1 for MASK class, + 1 for sigma

        atom_embedding = torch.nn.Linear(self.num_classes, hyper_params.atom_type_embedding_size)

        self.atom_type_output_layer = torch.nn.Linear(hyper_params.atom_type_embedding_size, self.num_classes)

        self.gemnet = GemNetT(
            num_targets=1,
            latent_dim=hyper_params.latent_dim,
            atom_embedding=atom_embedding,
            otf_graph=True,
            emb_size_atom=hyper_params.atom_type_embedding_size
        )

    @staticmethod
    def _create_block_diagonal_projection_matrices(
        bloch_wave_reciprocal_lattice_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Create block diagonal projection matrices.

        This method creates the "Gamma" matrices that are needed to project the higher dimensional
        outputs of EGNN back to the relevant dimension for the normalized score.

        The normalized score is defined as
            S = z . Gamma^alpha . hat_z
        where hat_z is the output of the EGNN model, z are the uplifted Euclidean position. This method
        creates the Gamma matrices, expressed as a tensor Gamma_{alpha, i, j}, where "alpha" is a coordinate
        in real space, and i,j are coordinates in the uplifted Euclidean space.

        Args:
            bloch_wave_reciprocal_lattice_vectors: K vectors, in reduced units.

        Returns:
            projection_matrices: block diagonal projection matrices, of
                dimension [spatial_dimension, uplifted Euclidean dimension, uplifted Euclidean dimension].
        """
        dimensional_projector = torch.tensor([[0.0, -1.0], [1.0, 0.0]])

        spatial_dimension = bloch_wave_reciprocal_lattice_vectors.shape[1]

        projection_matrices = []
        for alpha in range(spatial_dimension):
            list_k_alpha = bloch_wave_reciprocal_lattice_vectors[:, alpha]
            blocks = [k_alpha * dimensional_projector for k_alpha in list_k_alpha]
            projection_matrices.append(torch.block_diag(*blocks))

        return torch.stack(projection_matrices)

    def _forward_unchecked(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> AXL:
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        batch_size, number_of_atoms, spatial_dimension = relative_coordinates.shape

        frac_coords = relative_coordinates.view(-1, spatial_dimension)  # (N_atoms, 3)
        atom_types_one_hot = (
            torch.nn.functional.one_hot(batch[NOISY_AXL_COMPOSITION].A.view(-1), num_classes=self.num_classes).float()
        )  # (N_atoms, )
        num_atoms = torch.ones(batch_size).to(relative_coordinates.device).long() * number_of_atoms
        batch_indices = torch.arange(batch_size).repeat_interleave(number_of_atoms).to(relative_coordinates.device)

        raw_gemnet_output = self.gemnet(
            frac_coords=frac_coords,
            atom_types=atom_types_one_hot,
            num_atoms=num_atoms,
            batch=batch_indices,
            lattice=batch[UNIT_CELL]
        )

        normalized_scores = einops.rearrange(
            raw_gemnet_output.forces,
            "(batch natoms) spatial_dimension -> batch natoms spatial_dimension",
            batch=batch_size,
            natoms=number_of_atoms,
        )

        # atom_scores:
        atom_scores = self.atom_type_output_layer(raw_gemnet_output.node_embeddings)

        atom_reshaped_scores = einops.rearrange(
            atom_scores,
            "(batch natoms) num_classes -> batch natoms num_classes",
            batch=batch_size,
            natoms=number_of_atoms,
        )

        axl_scores = AXL(
            A=atom_reshaped_scores,
            X=normalized_scores,
            L=torch.zeros_like(normalized_scores),
        )

        return axl_scores

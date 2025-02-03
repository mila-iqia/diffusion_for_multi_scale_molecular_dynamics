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
from diffusion_for_multi_scale_molecular_dynamics.utils.lattice_utils import \
    get_cubic_point_group_positive_normalized_bloch_wave_vectors


@dataclass(kw_only=True)
class EGNNScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for ENN score networks."""

    architecture: str = "egnn"
    number_of_bloch_wave_shells: int = 1
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
    edges: str = "fully_connected"
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

        self.spatial_dimension = hyper_params.spatial_dimension
        self.num_atom_types = hyper_params.num_atom_types
        self.number_of_features_per_node = (
            self.num_atom_types + 2
        )  # +1 for MASK class, + 1 for sigma

        self.number_of_bloch_wave_shells = hyper_params.number_of_bloch_wave_shells
        bloch_wave_reciprocal_lattice_vectors = (
            get_cubic_point_group_positive_normalized_bloch_wave_vectors(
                number_of_complete_shells=self.number_of_bloch_wave_shells,
                spatial_dimension=self.spatial_dimension,
            )
        ).float()

        self.register_parameter(
            "bloch_wave_reciprocal_lattice_vectors",
            torch.nn.Parameter(
                bloch_wave_reciprocal_lattice_vectors, requires_grad=False
            ),
        )

        projection_matrices = self._create_block_diagonal_projection_matrices(
            bloch_wave_reciprocal_lattice_vectors
        )

        self.register_parameter(
            "projection_matrices",
            torch.nn.Parameter(projection_matrices, requires_grad=False),
        )

        self.edges = hyper_params.edges
        assert self.edges in [
            "fully_connected",
            "radial_cutoff",
        ], f"Edges type should be fully_connected or radial_cutoff. Got {self.edges}"

        self.radial_cutoff = hyper_params.radial_cutoff

        if self.edges == "fully_connected":
            assert (
                self.radial_cutoff is None
            ), "Specifying a radial cutoff is inconsistent with edges=fully_connected."
        else:
            assert (
                type(self.radial_cutoff) is float
            ), "A floating point value for the radial cutoff is needed for edges=radial_cutoff."

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
            num_classes=self.num_atom_types + 1,
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

    @staticmethod
    def _get_node_attributes(
        batch: Dict[AnyStr, torch.Tensor], num_atom_types: int
    ) -> torch.Tensor:
        """Get node attributes.

        This method extracts the node attributes, "h", to be fed as input to the EGNN network.
        Args:
            batch : the batch dictionary
            num_atom_types: number of atom types excluding the MASK token

        Returns:
            node_attributes: a tensor of dimension [batch, natoms, num_atom_types + 2]
        """
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        batch_size, number_of_atoms, spatial_dimension = relative_coordinates.shape

        sigmas = batch[NOISE].to(relative_coordinates.device)
        repeated_sigmas = einops.repeat(
            sigmas, "batch 1 -> (batch natoms) 1", natoms=number_of_atoms
        )

        atom_types = batch[NOISY_AXL_COMPOSITION].A
        atom_types_one_hot = class_index_to_onehot(
            atom_types, num_classes=num_atom_types + 1
        )

        node_attributes = torch.concatenate(
            (repeated_sigmas, atom_types_one_hot.view(-1, num_atom_types + 1)), dim=1
        )
        return node_attributes

    def _get_euclidean_positions(
        self,
        flat_relative_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Get Euclidean positions.

        Get the positions that take points on the torus into a higher dimensional
        (ie, [e^{iK r}]i) Euclidean-like space.

        Args:
            flat_relative_coordinates: relative coordinates, of dimensions [number_of_nodes, spatial_dimension]

        Returns:
            euclidean_positions : uplifted relative coordinates to a higher dimensional Euclidean space.
        """
        # Uplift the relative coordinates to the embedding Euclidean space
        two_pi_x = 2.0 * torch.pi * flat_relative_coordinates
        kr = einops.einsum(
            self.bloch_wave_reciprocal_lattice_vectors.to(two_pi_x),
            two_pi_x,
            "nbloch space, nodes space -> nodes nbloch",
        )
        cosines = kr.cos()
        sines = kr.sin()
        euclidean_positions = einops.rearrange(
            torch.stack([cosines, sines]),
            "two nodes nbloch -> nodes (nbloch two)",
        )
        return euclidean_positions

    def _forward_unchecked(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> AXL:
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        batch_size, number_of_atoms, spatial_dimension = relative_coordinates.shape

        if self.edges == "fully_connected":
            edges = get_edges_batch(n_nodes=number_of_atoms, batch_size=batch_size)
        else:
            edges = get_edges_with_radial_cutoff(
                relative_coordinates,
                batch[UNIT_CELL],
                self.radial_cutoff,
                drop_duplicate_edges=self.drop_duplicate_edges,
                spatial_dimension=self.spatial_dimension,
            )

        edges = edges.to(relative_coordinates.device)

        flat_relative_coordinates = einops.rearrange(
            relative_coordinates,
            "batch natom spatial_dimension -> (batch natom) spatial_dimension",
        )

        # Uplift the relative coordinates to the embedding Euclidean space.
        #   Dimensions [number_of_nodes, 2 x spatial_dimension]
        euclidean_positions = self._get_euclidean_positions(flat_relative_coordinates)

        node_attributes_h = self._get_node_attributes(
            batch, num_atom_types=self.num_atom_types
        )
        # The raw normalized score has dimensions [number_of_nodes, 2 x spatial_dimension]
        # CAREFUL! It is important to pass a clone of the euclidian positions because EGNN will modify its input!
        raw_normalized_score = self.egnn(
            h=node_attributes_h, edges=edges, x=euclidean_positions.clone()
        )

        # The projected score is defined a
        #       S^alpha = z . Gamma^alpha . hat_z
        #  where:
        #       - alpha is a spatial index (ie, x, y z) in the real space
        #       - z are the "positions" in the uplifted Euclidean space
        #       - hat_z is the output of the EGNN model, also in the uplifted Euclidean space
        #       - Gamma^alpha are the projection matrices
        flat_normalized_scores = einops.einsum(
            euclidean_positions,
            self.projection_matrices,
            raw_normalized_score.X,
            "nodes i, alpha i j, nodes j-> nodes alpha",
        )

        normalized_scores = einops.rearrange(
            flat_normalized_scores,
            "(batch natoms) spatial_dimension -> batch natoms spatial_dimension",
            batch=batch_size,
            natoms=number_of_atoms,
        )

        atom_reshaped_scores = einops.rearrange(
            raw_normalized_score.A,
            "(batch natoms) num_classes -> batch natoms num_classes",
            batch=batch_size,
            natoms=number_of_atoms,
        )

        axl_scores = AXL(
            A=atom_reshaped_scores,
            X=normalized_scores,
            L=raw_normalized_score.L,
        )

        return axl_scores

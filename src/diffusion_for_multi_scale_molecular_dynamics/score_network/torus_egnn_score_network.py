from dataclasses import dataclass
from typing import Optional, Tuple

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn_score_network import (
    EGNNScoreNetwork, EGNNScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn_utils import (
    build_graph_periodic_edges_batch, minimum_image_cartesian)
from diffusion_for_multi_scale_molecular_dynamics.score_network.torus_egnn import \
    TorusEGNN
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates,
    map_lattice_parameters_to_unit_cell_vectors)
from diffusion_for_multi_scale_molecular_dynamics.utils.lattice_utils import \
    get_cubic_point_group_positive_normalized_bloch_wave_vectors


@dataclass(kw_only=True)
class TorusEGNNScoreNetworkParameters(EGNNScoreNetworkParameters):
    """Hyper-parameters of the EGNN score network on the torus."""

    architecture: str = "egnn_torus"
    number_of_bloch_wave_shells: int = 1
    edges: str = "radial_cutoff"

    def __post_init__(self):
        """Post init checks."""
        super().__post_init__()
        assert self.normalize in ("raw", "tanh"), \
            f"normalize should be raw or tanh on the torus, where distances are not in Angstrom. Got {self.normalize}"
        assert self.edges in ("radial_cutoff", "fully_connected"), \
            f"edges should be radial_cutoff or fully_connected. Got {self.edges}"
        if self.edges == "fully_connected":
            assert self.radial_cutoff is None and not self.smooth_cutoff, \
                "fully_connected edges need radial_cutoff=None and smooth_cutoff=False."
        else:
            assert type(self.radial_cutoff) is float, "A floating point value is needed for radial_cutoff."


class TorusEGNNScoreNetwork(EGNNScoreNetwork):
    """EGNN score network on the relative coordinates embedded on the torus.

    The relative coordinates are uplifted to the Euclidean positions z = [cos(2 pi K.x), sin(2 pi K.x)] of the Bloch
    wave vectors K, and EGNN's output is projected back to the score. Every pair of atoms is an edge at most once, at
    the cartesian distance of its closest periodic image: with edges="radial_cutoff", the pairs within the radial
    cutoff; with edges="fully_connected", every pair of distinct atoms.
    """

    def __init__(self, hyper_params: TorusEGNNScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(TorusEGNNScoreNetwork, self).__init__(hyper_params)

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
        self.egnn = TorusEGNN(**self._get_egnn_arguments(hyper_params))

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

    def _get_closest_image_edges(
        self,
        cartesian_coordinates: torch.Tensor,
        cell_lengths: torch.Tensor,
        natoms: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get one edge per pair of distinct atoms within the radial cutoff, at the distance of its closest image.

        On the torus, every periodic image of a pair gives the same Euclidean positions, so each pair is kept once.

        Args:
            cartesian_coordinates: atomic positions. size: batch_size, max_atoms, spatial_dimension
            cell_lengths: per-sample orthogonal cell side lengths. size: batch_size, spatial_dimension
            natoms: optional real (non-padded) atom count per sample. size: batch_size

        Returns:
            edges: edges as [i, j], into the flattened (batch_size * max_atoms) node numbering.
                size: number of edges, 2
            edge_distances_squared: cartesian distance squared of the closest image of every edge (Angstrom^2).
                size: number of edges, 1
        """
        periodic_edges = build_graph_periodic_edges_batch(
            cartesian_coordinates, cell_lengths, self.radial_cutoff, natoms=natoms
        )
        flat_cartesian_coordinates = einops.rearrange(
            cartesian_coordinates, "batch natom spatial_dimension -> (batch natom) spatial_dimension"
        )
        row, col = periodic_edges[:, 0].long(), periodic_edges[:, 1].long()
        distances_squared = (
            (flat_cartesian_coordinates[row] - flat_cartesian_coordinates[col] - periodic_edges[:, 2:]) ** 2
        ).sum(dim=1)

        distinct_atoms = row != col
        row, col, distances_squared = row[distinct_atoms], col[distinct_atoms], distances_squared[distinct_atoms]

        number_of_nodes = flat_cartesian_coordinates.shape[0]
        pair_ids, pair_index_of_edge = torch.unique(row * number_of_nodes + col, return_inverse=True)
        closest_distances_squared = torch.full_like(pair_ids, torch.inf, dtype=distances_squared.dtype).scatter_reduce(
            0, pair_index_of_edge, distances_squared, reduce="amin"
        )
        edges = torch.stack([pair_ids // number_of_nodes, pair_ids % number_of_nodes], dim=1).float()
        return edges, closest_distances_squared.unsqueeze(1)

    @staticmethod
    def _get_fully_connected_edges(
        cartesian_coordinates: torch.Tensor,
        cell_lengths: torch.Tensor,
        natoms: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get one edge per pair of distinct atoms of every sample, at the distance of its closest image.

        Args:
            cartesian_coordinates: atomic positions. size: batch_size, max_atoms, spatial_dimension
            cell_lengths: per-sample orthogonal cell side lengths. size: batch_size, spatial_dimension
            natoms: optional real (non-padded) atom count per sample. size: batch_size

        Returns:
            edges: edges as [i, j], into the flattened (batch_size * max_atoms) node numbering.
                size: number of edges, 2
            edge_distances_squared: cartesian distance squared of the closest image of every edge (Angstrom^2).
                size: number of edges, 1
        """
        batch_size, max_atoms, _ = cartesian_coordinates.shape
        atom_indices = torch.arange(max_atoms, device=cartesian_coordinates.device)
        row, col = torch.meshgrid(atom_indices, atom_indices, indexing="ij")
        distinct_atoms = row != col
        row, col = row[distinct_atoms].repeat(batch_size), col[distinct_atoms].repeat(batch_size)
        batch_index = torch.arange(batch_size, device=cartesian_coordinates.device).repeat_interleave(
            max_atoms * (max_atoms - 1)
        )

        if natoms is not None:
            real_atoms = (row < natoms[batch_index]) & (col < natoms[batch_index])
            row, col, batch_index = row[real_atoms], col[real_atoms], batch_index[real_atoms]

        displacements = minimum_image_cartesian(
            cartesian_coordinates[batch_index, row] - cartesian_coordinates[batch_index, col], cell_lengths[batch_index]
        )
        edges = torch.stack([row + batch_index * max_atoms, col + batch_index * max_atoms], dim=1).float()
        return edges, (displacements**2).sum(dim=1, keepdim=True)

    def _get_flat_scores(
        self,
        node_attributes: torch.Tensor,
        relative_coordinates: torch.Tensor,
        lattice_parameters: torch.Tensor,
        natoms: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the scores of every atom from EGNN on the torus (see EGNNScoreNetwork)."""
        spatial_dimension = relative_coordinates.shape[-1]
        unit_cell = map_lattice_parameters_to_unit_cell_vectors(lattice_parameters)
        cartesian_coordinates = get_positions_from_coordinates(relative_coordinates, unit_cell)
        cell_lengths = lattice_parameters[:, :spatial_dimension]
        if self.edges == "fully_connected":
            edges, edge_distances_squared = self._get_fully_connected_edges(cartesian_coordinates, cell_lengths, natoms)
        else:
            edges, edge_distances_squared = self._get_closest_image_edges(cartesian_coordinates, cell_lengths, natoms)

        flat_relative_coordinates = einops.rearrange(
            relative_coordinates,
            "batch natom spatial_dimension -> (batch natom) spatial_dimension",
        )

        # Uplift the relative coordinates to the embedding Euclidean space.
        #   Dimensions [number_of_nodes, 2 x number of Bloch wave vectors]
        euclidean_positions = self._get_euclidean_positions(flat_relative_coordinates)

        # CAREFUL! It is important to pass a clone of the euclidian positions because EGNN will modify its input!
        raw_score = self.egnn(
            h=node_attributes, x=euclidean_positions.clone(), edges=edges,
            edge_distances_squared=edge_distances_squared,
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
            raw_score.X,
            "nodes i, alpha i j, nodes j-> nodes alpha",
        )
        return flat_normalized_scores, raw_score.A

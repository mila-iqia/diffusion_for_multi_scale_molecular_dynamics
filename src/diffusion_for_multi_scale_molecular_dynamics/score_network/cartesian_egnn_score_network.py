from dataclasses import dataclass
from typing import Tuple

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.score_network.cartesian_egnn import \
    CartesianEGNN
from diffusion_for_multi_scale_molecular_dynamics.score_network.egnn_score_network import (
    EGNNScoreNetwork, EGNNScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates,
    map_lattice_parameters_to_unit_cell_vectors)


@dataclass(kw_only=True)
class CartesianEGNNScoreNetworkParameters(EGNNScoreNetworkParameters):
    """Hyper-parameters of the EGNN score network on cartesian positions."""

    architecture: str = "egnn_cartesian"
    normalization_length: float = 0.1  # Angstrom
    rebuild_edges_every_layer: bool = True
    bond_direction_feature: bool = True

    def __post_init__(self):
        """Post init checks."""
        super().__post_init__()
        assert type(self.radial_cutoff) is float, "A floating point value is needed for radial_cutoff."


class CartesianEGNNScoreNetwork(EGNNScoreNetwork):
    """EGNN score network on the cartesian positions of the atoms.

    The edges are the periodic edges within the radial cutoff, and the score is the displacement of the positions by
    the EGNN coordinate updates.
    """

    def __init__(self, hyper_params: CartesianEGNNScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(CartesianEGNNScoreNetwork, self).__init__(hyper_params)
        self.egnn = CartesianEGNN(
            **self._get_egnn_arguments(hyper_params),
            normalization_length=hyper_params.normalization_length,
            rebuild_edges_every_layer=hyper_params.rebuild_edges_every_layer,
            bond_direction_feature=hyper_params.bond_direction_feature,
        )

    def _get_flat_scores(
        self,
        node_attributes: torch.Tensor,
        relative_coordinates: torch.Tensor,
        lattice_parameters: torch.Tensor,
        natoms: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the scores of every atom from EGNN on the cartesian positions (see EGNNScoreNetwork)."""
        spatial_dimension = relative_coordinates.shape[-1]
        unit_cell = map_lattice_parameters_to_unit_cell_vectors(lattice_parameters)
        cartesian_coordinates = get_positions_from_coordinates(relative_coordinates, unit_cell)

        flat_cartesian_coordinates = einops.rearrange(
            cartesian_coordinates,
            "batch natom spatial_dimension -> (batch natom) spatial_dimension",
        )

        # CAREFUL! It is important to pass a clone of the cartesian positions because EGNN will modify its input!
        raw_score = self.egnn(
            h=node_attributes, x=flat_cartesian_coordinates.clone(),
            cell_lengths=lattice_parameters[:, :spatial_dimension], natoms=natoms,
        )

        # EGNN's coordinate branch returns an updated position; its displacement is the sigma-normalized
        # score sigma_cart * grad_cart log p = sigma_rel * grad_rel log p, which is the same in both frames.
        return raw_score.X - flat_cartesian_coordinates, raw_score.A

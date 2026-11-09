from dataclasses import dataclass
from typing import Any, AnyStr, Dict, Optional, Tuple

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISE, NOISY_AXL_COMPOSITION, NUMBER_OF_ATOMS)
from diffusion_for_multi_scale_molecular_dynamics.score_network import \
    ScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.score_network.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    class_index_to_onehot


@dataclass(kw_only=True)
class EGNNScoreNetworkParameters(ScoreNetworkParameters):
    """Hyper-parameters shared by the EGNN score networks."""

    message_n_hidden_dimensions: int = 1
    message_hidden_dimensions_size: int = 16
    node_n_hidden_dimensions: int = 1
    node_hidden_dimensions_size: int = 32
    coordinate_n_hidden_dimensions: int = 1
    coordinate_hidden_dimensions_size: int = 32
    residual: bool = True
    attention: bool = False
    normalize: str = "raw"
    tanh: bool = False
    coords_agg: str = "mean"
    message_agg: str = "mean"
    n_layers: int = 4
    radial_cutoff: Optional[float]  # Angstrom
    smooth_cutoff: bool = True
    supports_variable_natoms: bool = True
    noise_input: str = "log"


class EGNNScoreNetwork(ScoreNetwork):
    """Base class of the score networks using EGNN.

    This implementation leverages the baseline EGNN architecture from Satorras et al.
    Supports variable numbers of atoms across samples in the same batch via padding.
    The subclasses define how the atoms are given to EGNN and how its output becomes the score.
    """

    def __init__(self, hyper_params: EGNNScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(EGNNScoreNetwork, self).__init__(hyper_params)

        self.spatial_dimension = hyper_params.spatial_dimension
        self.num_atom_types = hyper_params.num_atom_types
        assert hyper_params.noise_input in ("log", "raw"), f"unknown noise_input {hyper_params.noise_input}"
        self.noise_input = hyper_params.noise_input
        self.number_of_features_per_node = (
            self.num_atom_types + 2
        )  # +1 for MASK class, + 1 for sigma
        self.radial_cutoff = hyper_params.radial_cutoff

    def _get_egnn_arguments(self, hyper_params: EGNNScoreNetworkParameters) -> Dict[str, Any]:
        """Get the EGNN arguments shared by the EGNN score networks.

        Args:
            hyper_params : hyper parameters from the config file.

        Returns:
            egnn_arguments: the shared arguments of EGNN.
        """
        return dict(
            input_size=self.number_of_features_per_node,
            message_n_hidden_dimensions=hyper_params.message_n_hidden_dimensions,
            message_hidden_dimensions_size=hyper_params.message_hidden_dimensions_size,
            node_n_hidden_dimensions=hyper_params.node_n_hidden_dimensions,
            node_hidden_dimensions_size=hyper_params.node_hidden_dimensions_size,
            coordinate_n_hidden_dimensions=hyper_params.coordinate_n_hidden_dimensions,
            coordinate_hidden_dimensions_size=hyper_params.coordinate_hidden_dimensions_size,
            radial_cutoff=self.radial_cutoff,
            residual=hyper_params.residual,
            attention=hyper_params.attention,
            normalize=hyper_params.normalize,
            tanh=hyper_params.tanh,
            coords_agg=hyper_params.coords_agg,
            message_agg=hyper_params.message_agg,
            n_layers=hyper_params.n_layers,
            smooth_cutoff=hyper_params.smooth_cutoff,
            num_classes=self.num_atom_types + 1,
        )

    @staticmethod
    def _get_node_attributes(
        batch: Dict[AnyStr, torch.Tensor], num_atom_types: int, noise_input: str = "log"
    ) -> torch.Tensor:
        """Get node attributes.

        This method extracts the node attributes, "h", to be fed as input to the EGNN network.
        Args:
            batch : the batch dictionary
            num_atom_types: number of atom types excluding the MASK token
            noise_input: "log" to give ln sigma to the network, "raw" to give sigma itself. Defaults to "log".

        Returns:
            node_attributes: a tensor of dimension [batch, natoms, num_atom_types + 2]
        """
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        batch_size, max_atoms, spatial_dimension = relative_coordinates.shape

        sigmas = batch[NOISE].to(relative_coordinates.device)
        if noise_input == "log":
            sigmas = torch.log(sigmas)
        repeated_sigmas = einops.repeat(
            sigmas, "batch 1 -> (batch natoms) 1", natoms=max_atoms
        )

        atom_types = batch[NOISY_AXL_COMPOSITION].A.clamp(min=0)  # Cannot have -1 padding in onehot
        atom_types_one_hot = class_index_to_onehot(
            atom_types, num_classes=num_atom_types + 1
        )

        node_attributes = torch.concatenate(
            (repeated_sigmas, atom_types_one_hot.view(-1, num_atom_types + 1)), dim=1
        )
        return node_attributes

    def _get_flat_scores(
        self,
        node_attributes: torch.Tensor,
        relative_coordinates: torch.Tensor,
        lattice_parameters: torch.Tensor,
        natoms: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the scores of every atom from EGNN.

        Args:
            node_attributes: node attributes. size: batch_size * max_atoms, num_atom_types + 2
            relative_coordinates: noisy relative coordinates. size: batch_size, max_atoms, spatial_dimension
            lattice_parameters: lattice parameters of orthogonal cells. size: batch_size, number of lattice parameters
            natoms: real (non-padded) atom count per sample. size: batch_size

        Returns:
            normalized_scores: sigma-normalized coordinates scores. size: batch_size * max_atoms, spatial_dimension
            atom_type_logits: atom type logits. size: batch_size * max_atoms, num_atom_types + 1
        """
        raise NotImplementedError

    def _forward_unchecked(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> AXL:
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        batch_size, max_atoms, spatial_dimension = relative_coordinates.shape

        lattice_parameters = batch[NOISY_AXL_COMPOSITION].L
        lattice_parameters[:, spatial_dimension:] = 0  # TODO force orthogonal cell

        node_attributes_h = self._get_node_attributes(
            batch, num_atom_types=self.num_atom_types, noise_input=self.noise_input
        )
        flat_normalized_scores, flat_atom_type_logits = self._get_flat_scores(
            node_attributes_h, relative_coordinates, lattice_parameters, batch[NUMBER_OF_ATOMS]
        )

        normalized_scores = einops.rearrange(
            flat_normalized_scores,
            "(batch natoms) spatial_dimension -> batch natoms spatial_dimension",
            batch=batch_size,
            natoms=max_atoms,
        )

        atom_reshaped_scores = einops.rearrange(
            flat_atom_type_logits,
            "(batch natoms) num_classes -> batch natoms num_classes",
            batch=batch_size,
            natoms=max_atoms,
        )

        axl_scores = AXL(
            A=atom_reshaped_scores,
            X=normalized_scores,
            L=torch.zeros_like(batch[NOISY_AXL_COMPOSITION].L),
        )
        return axl_scores

from dataclasses import dataclass, field
from typing import AnyStr, Dict, List

import torch
from e3nn import o3
from mace.modules import gate_dict, interaction_classes
from mace.tools.torch_geometric.dataloader import Collater

from crystal_diffusion.models.diffusion_mace import (DiffusionMACE,
                                                     input_to_diffusion_mace)
from crystal_diffusion.models.score_networks.score_network import (
    ScoreNetwork, ScoreNetworkParameters)
from crystal_diffusion.namespace import (NOISY_CARTESIAN_POSITIONS,
                                         NOISY_RELATIVE_COORDINATES, UNIT_CELL)
from crystal_diffusion.utils.basis_transformations import (
    get_positions_from_coordinates, get_reciprocal_basis_vectors)


@dataclass(kw_only=True)
class DiffusionMACEScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for Diffusion MACE score networks."""
    architecture: str = 'diffusion_mace'
    number_of_atoms: int  # the number of atoms in a configuration.
    number_of_elements: int = 1  # The number of distinct elements present
    r_max: float = 5.0
    num_bessel: int = 8
    num_polynomial_cutoff: int = 5
    max_ell: int = 2
    interaction_cls: str = "RealAgnosticResidualInteractionBlock"
    interaction_cls_first: str = "RealAgnosticInteractionBlock"
    num_interactions: int = 2
    hidden_irreps: str = "128x0e + 128x1o"  # irreps for hidden node states
    mlp_irreps: str = "16x0e"  # irreps for the embedding of the diffusion scalar features
    number_of_mlp_layers: int = 3  # number of MLP layers for the embedding of the diffusion scalar features
    avg_num_neighbors: int = 1  # normalization factor for the message
    correlation: int = 3
    gate: str = "silu"  # non linearity for last readout - choices: ["silu", "tanh", "abs", "None"]
    radial_MLP: List[int] = field(default_factory=lambda: [64, 64, 64])  # "width of the radial MLP"
    radial_type: str = "bessel"  # type of radial basis functions - choices=["bessel", "gaussian", "chebyshev"]
    condition_embedding_size: int = 64  # dimension of the conditional variable embedding - assumed to be l=1 (odd)


class DiffusionMACEScoreNetwork(ScoreNetwork):
    """Score network using Diffusion MACE."""

    def __init__(self, hyper_params: DiffusionMACEScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(DiffusionMACEScoreNetwork, self).__init__(hyper_params)

        # dataloader
        self.r_max = hyper_params.r_max
        self.collate_fn = Collater(follow_batch=[None], exclude_keys=[None])

        diffusion_mace_config = dict(
            r_max=hyper_params.r_max,
            num_bessel=hyper_params.num_bessel,
            num_polynomial_cutoff=hyper_params.num_polynomial_cutoff,
            max_ell=hyper_params.max_ell,
            interaction_cls=interaction_classes[hyper_params.interaction_cls],
            interaction_cls_first=interaction_classes[hyper_params.interaction_cls_first],
            num_interactions=hyper_params.num_interactions,
            num_elements=hyper_params.number_of_elements,
            hidden_irreps=o3.Irreps(hyper_params.hidden_irreps),
            mlp_irreps=o3.Irreps(hyper_params.mlp_irreps),
            number_of_mlp_layers=hyper_params.number_of_mlp_layers,
            avg_num_neighbors=hyper_params.avg_num_neighbors,
            atomic_numbers=[14],  # TODO: revisit this when we have multi-atom types
            correlation=hyper_params.correlation,
            gate=gate_dict[hyper_params.gate],
            radial_MLP=hyper_params.radial_MLP,
            radial_type=hyper_params.radial_type,
            condition_embedding_size=hyper_params.condition_embedding_size,
        )

        self._natoms = hyper_params.number_of_atoms
        self._number_of_elements = hyper_params.number_of_elements

        self.diffusion_mace_network = DiffusionMACE(**diffusion_mace_config)

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        super(DiffusionMACEScoreNetwork, self)._check_batch(batch)
        number_of_atoms = batch[NOISY_RELATIVE_COORDINATES].shape[1]
        assert (
            number_of_atoms == self._natoms
        ), "The dimension corresponding to the number of atoms is not consistent with the configuration."

    def _forward_unchecked(self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False) -> torch.Tensor:
        """Forward unchecked.

        This method assumes that the input data has already been checked with respect to expectations
        and computes the scores assuming that the data is in the correct format.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional (optional): if True, do a forward as though the model was conditional on the forces.
                Defaults to False.

        Returns:
            output : the scores computed by the model as a [batch_size, n_atom, spatial_dimension] tensor.
        """
        relative_coordinates = batch[NOISY_RELATIVE_COORDINATES]
        batch_size, number_of_atoms, spatial_dimension = relative_coordinates.shape

        basis_vectors = batch[UNIT_CELL]
        batch[NOISY_CARTESIAN_POSITIONS] = get_positions_from_coordinates(relative_coordinates, basis_vectors)
        graph_input = input_to_diffusion_mace(batch, radial_cutoff=self.r_max)

        flat_cartesian_scores = self.diffusion_mace_network(graph_input, conditional)
        cartesian_scores = flat_cartesian_scores.reshape(batch_size, number_of_atoms, spatial_dimension)

        reciprocal_basis_vectors_as_columns = get_reciprocal_basis_vectors(basis_vectors)
        scores = torch.bmm(cartesian_scores, reciprocal_basis_vectors_as_columns)

        return scores

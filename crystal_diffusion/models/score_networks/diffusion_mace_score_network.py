from dataclasses import dataclass, field
from typing import AnyStr, Dict, List

import numpy as np
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
    prediction_head: str = 'non_conservative'
    number_of_atoms: int  # the number of atoms in a configuration.
    r_max: float = 5.0
    num_bessel: int = 8
    num_polynomial_cutoff: int = 5
    max_ell: int = 2
    interaction_cls: str = "RealAgnosticResidualInteractionBlock"
    interaction_cls_first: str = "RealAgnosticInteractionBlock"
    num_interactions: int = 2
    hidden_irreps: str = "128x0e + 128x1o"  # irreps for hidden node states
    MLP_irreps: str = "16x0e"  # from mace.tools.arg_parser
    avg_num_neighbors: int = 1  # normalization factor for the message
    correlation: int = 3
    gate: str = "silu"  # non linearity for last readout - choices: ["silu", "tanh", "abs", "None"]
    radial_MLP: List[int] = field(default_factory=lambda: [64, 64, 64])  # "width of the radial MLP"
    radial_type: str = "bessel"  # type of radial basis functions - choices=["bessel", "gaussian", "chebyshev"]


class DiffusionMACEScoreNetwork(ScoreNetwork):
    """Score network using Diffusion MACE."""

    def __init__(self, hyper_params: DiffusionMACEScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(DiffusionMACEScoreNetwork, self).__init__(hyper_params)

        assert hyper_params.prediction_head in ['non_conservative', 'energy_gradient'], \
            f"unknown prediction head '{hyper_params.prediction_head}'"

        self.prediction_head = hyper_params.prediction_head

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
            num_elements=1,  # TODO: revisit this when we have multi-atom types
            hidden_irreps=o3.Irreps(hyper_params.hidden_irreps),
            MLP_irreps=o3.Irreps(hyper_params.MLP_irreps),
            atomic_energies=np.array([0.]),
            avg_num_neighbors=hyper_params.avg_num_neighbors,
            atomic_numbers=[14],  # TODO: revisit this when we have multi-atom types
            correlation=hyper_params.correlation,
            gate=gate_dict[hyper_params.gate],
            radial_MLP=hyper_params.radial_MLP,
            radial_type=hyper_params.radial_type
        )

        self._natoms = hyper_params.number_of_atoms

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
        del conditional  # TODO do something with forces when conditional
        relative_coordinates = batch[NOISY_RELATIVE_COORDINATES]
        batch_size, number_of_atoms, spatial_dimension = relative_coordinates.shape

        basis_vectors = batch[UNIT_CELL]
        batch[NOISY_CARTESIAN_POSITIONS] = get_positions_from_coordinates(relative_coordinates, basis_vectors)
        graph_input = input_to_diffusion_mace(batch, radial_cutoff=self.r_max)

        if self.prediction_head == 'energy_gradient':
            compute_force = True
        else:
            compute_force = False

        diffusion_mace_output = self.diffusion_mace_network(graph_input,
                                                            compute_force=compute_force,
                                                            training=self.training)

        flat_cartesian_scores = diffusion_mace_output[self.prediction_head]

        cartesian_scores = flat_cartesian_scores.reshape(batch_size, number_of_atoms, spatial_dimension)

        if self.prediction_head == 'energy_gradient':
            # Using the chain rule, we can derive nabla_x given nabla_r, where 'x' is relative coordinates and 'r'
            # is cartesian space.
            scores = torch.bmm(cartesian_scores, basis_vectors.transpose(2, 1))
        elif self.prediction_head == 'non_conservative':
            reciprocal_basis_vectors_as_columns = get_reciprocal_basis_vectors(basis_vectors)
            scores = torch.bmm(cartesian_scores, reciprocal_basis_vectors_as_columns)
        else:
            raise ValueError(f"Unknown prediction head '{self.prediction_head}'")

        return scores

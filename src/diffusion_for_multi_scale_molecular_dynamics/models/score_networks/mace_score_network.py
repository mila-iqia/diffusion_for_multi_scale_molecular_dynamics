from dataclasses import dataclass, field
from typing import AnyStr, Dict, List, Optional

import numpy as np
import torch
from e3nn import o3
from mace.modules import MACE, gate_dict, interaction_classes
from mace.tools import get_atomic_number_table_from_zs
from mace.tools.torch_geometric.dataloader import Collater

from diffusion_for_multi_scale_molecular_dynamics.models.mace_utils import (
    build_mace_output_nodes_irreducible_representation, get_pretrained_mace,
    get_pretrained_mace_output_node_features_irreps, input_to_mace)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import (
    ScoreNetwork, ScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_prediction_head import (
    MaceScorePredictionHeadParameters, instantiate_mace_prediction_head)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    NOISY_CARTESIAN_POSITIONS, NOISY_RELATIVE_COORDINATES, TIME, UNIT_CELL)


@dataclass(kw_only=True)
class MACEScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for MACE score networks."""

    architecture: str = "mace"
    number_of_atoms: int  # the number of atoms in a configuration.
    use_pretrained: Optional[str] = (
        None  # if None, do not use pre-trained ; else, should be in small, medium or large
    )
    pretrained_weights_path: str = "../"  # path to pre-trained model weights
    r_max: float = 5.0
    num_bessel: int = 8
    num_polynomial_cutoff: int = 5
    max_ell: int = 2
    interaction_cls: str = "RealAgnosticResidualInteractionBlock"
    interaction_cls_first: str = "RealAgnosticInteractionBlock"
    num_interactions: int = 2
    hidden_irreps: str = "128x0e + 128x1o"  # irreps for hidden node states
    MLP_irreps: str = "16x0e"  # from mace.tools.arg_parser
    atomic_energies: np.ndarray = field(default_factory=lambda: np.zeros((89,)))
    avg_num_neighbors: int = 1  # normalization factor for the message
    correlation: int = 3
    gate: str = (
        "silu"  # non linearity for last readout - choices: ["silu", "tanh", "abs", "None"]
    )
    radial_MLP: List[int] = field(
        default_factory=lambda: [64, 64, 64]
    )  # "width of the radial MLP"
    radial_type: str = (
        "bessel"  # type of radial basis functions - choices=["bessel", "gaussian", "chebyshev"]
    )
    prediction_head_parameters: MaceScorePredictionHeadParameters


class MACEScoreNetwork(ScoreNetwork):
    """Score network using atom features from MACE.

    Inherits from the given framework's model class.
    """

    def __init__(self, hyper_params: MACEScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(MACEScoreNetwork, self).__init__(hyper_params)

        self.z_table = get_atomic_number_table_from_zs(
            list(range(89))
        )  # need 89 for pre-trained model
        # dataloader
        self.r_max = hyper_params.r_max
        self.collate_fn = Collater(follow_batch=[None], exclude_keys=[None])

        mace_config = dict(
            r_max=hyper_params.r_max,
            num_bessel=hyper_params.num_bessel,
            num_polynomial_cutoff=hyper_params.num_polynomial_cutoff,
            max_ell=hyper_params.max_ell,
            interaction_cls=interaction_classes[hyper_params.interaction_cls],
            interaction_cls_first=interaction_classes[
                hyper_params.interaction_cls_first
            ],
            num_interactions=hyper_params.num_interactions,
            num_elements=len(self.z_table),
            hidden_irreps=o3.Irreps(hyper_params.hidden_irreps),
            MLP_irreps=o3.Irreps(hyper_params.MLP_irreps),
            atomic_energies=hyper_params.atomic_energies,
            avg_num_neighbors=hyper_params.avg_num_neighbors,
            atomic_numbers=self.z_table.zs,
            correlation=hyper_params.correlation,
            gate=gate_dict[hyper_params.gate],
            radial_MLP=hyper_params.radial_MLP,
            radial_type=hyper_params.radial_type,
        )

        self._natoms = hyper_params.number_of_atoms

        if hyper_params.use_pretrained is None or hyper_params.use_pretrained == "None":
            self.mace_network = MACE(**mace_config)
            output_node_features_irreps = (
                build_mace_output_nodes_irreducible_representation(
                    hyper_params.hidden_irreps, hyper_params.num_interactions
                )
            )
        else:
            output_node_features_irreps = (
                get_pretrained_mace_output_node_features_irreps(
                    hyper_params.use_pretrained
                )
            )
            self.mace_network, mace_output_size = get_pretrained_mace(
                hyper_params.use_pretrained, hyper_params.pretrained_weights_path
            )
            assert (
                output_node_features_irreps.dim == mace_output_size
            ), "Something is wrong with pretrained dimensions."

        self.mace_output_size = output_node_features_irreps.dim
        self.prediction_head = instantiate_mace_prediction_head(
            output_node_features_irreps, hyper_params.prediction_head_parameters
        )

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        super(MACEScoreNetwork, self)._check_batch(batch)
        number_of_atoms = batch[NOISY_RELATIVE_COORDINATES].shape[1]
        assert (
            number_of_atoms == self._natoms
        ), "The dimension corresponding to the number of atoms is not consistent with the configuration."

    def _forward_unchecked(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> torch.Tensor:
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
        del conditional  # TODO implement conditional
        relative_coordinates = batch[NOISY_RELATIVE_COORDINATES]
        batch[NOISY_CARTESIAN_POSITIONS] = torch.bmm(
            relative_coordinates, batch[UNIT_CELL]
        )  # positions in Angstrom
        graph_input = input_to_mace(batch, radial_cutoff=self.r_max)
        mace_output = self.mace_network(
            graph_input, compute_force=False, training=self.training
        )

        # The node features are organized as (batchsize * natoms, output_size) in the mace output because
        # torch_geometric puts all the graphs in a batch in a single large graph.
        flat_node_features = mace_output["node_feats"]

        # The times have a single value per batch element; we repeat the array so that there is one value per atom,
        # with this value the same for all atoms belonging to the same graph.
        times = batch[TIME].to(relative_coordinates.device)  # shape [batch_size, 1]
        flat_times = times[graph_input.batch]  # shape [batch_size * natoms, 1]
        flat_scores = self.prediction_head(
            flat_node_features, flat_times
        )  # shape [batch_size * natoms, spatial_dim]

        # Reshape the scores to have an explicit batch dimension
        scores = flat_scores.reshape(-1, self._natoms, self.spatial_dimension)

        return scores

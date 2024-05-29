"""Score Network.

This module implements score networks for positions in relative coordinates.
Relative coordinates are with respect to lattice vectors which define the
periodic unit cell.
"""
import os
from dataclasses import dataclass, field
from typing import AnyStr, Dict, List, Optional

import numpy as np
import torch
from e3nn import o3
from mace.modules import gate_dict, interaction_classes
from mace.modules.models import MACE
from mace.tools import get_atomic_number_table_from_zs
from mace.tools.torch_geometric.dataloader import Collater
from torch import nn

from crystal_diffusion.models.diffusion_mace import (DiffusionMACE,
                                                     input_to_diffusion_mace)
from crystal_diffusion.models.mace_utils import (
    build_mace_output_nodes_irreducible_representation, get_pretrained_mace,
    get_pretrained_mace_output_node_features_irreps, input_to_mace)
from crystal_diffusion.models.score_prediction_head import (
    MaceScorePredictionHeadParameters, instantiate_mace_prediction_head)
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_CARTESIAN_POSITIONS,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)
from crystal_diffusion.utils.basis_transformations import \
    get_positions_from_coordinates

# mac fun time
# for mace, conflict with mac
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already- \
# initial
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@dataclass(kw_only=True)
class ScoreNetworkParameters:
    """Base Hyper-parameters for score networks."""
    architecture: str
    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live.
    conditional_prob: float = 0.  # probability of making an conditional forward - else, do a unconditional forward
    conditional_gamma: float = 2.  # conditional score weighting - see eq. B45 in MatterGen
    # p_\gamma(x|c) = p(c|x)^\gamma p(x)


class ScoreNetwork(torch.nn.Module):
    """Base score network.

    This base class defines the interface that all score networks should have
    in order to be easily interchangeable (ie, polymorphic).
    """

    def __init__(self, hyper_params: ScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyperparameters from the config file.
        """
        super(ScoreNetwork, self).__init__()
        self._hyper_params = hyper_params
        self.spatial_dimension = hyper_params.spatial_dimension
        self.conditional_prob = hyper_params.conditional_prob
        self.conditional_gamma = hyper_params.conditional_gamma

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        """Check batch.

        Check that the batch dictionary contains the expected inputs, and that
        those inputs have the expected dimensions.

        It is expected that:
            - the relative coordinates are present and of shape [batch_size, number of atoms, spatial_dimension]
            - all the components of relative coordinates will be in [0, 1)
            - the time steps are present and of shape [batch_size, 1]
            - the time steps are in range [0, 1].
            - the 'noise' parameter is present and has the same shape as time.

        An assert will fail if the batch does not conform with expectation.

        Args:
            batch : dictionary containing the data to be processed by the model.

        Returns:
            None.
        """
        assert NOISY_RELATIVE_COORDINATES in batch, \
            (f"The relative coordinates should be present in "
             f"the batch dictionary with key '{NOISY_RELATIVE_COORDINATES}'")

        relative_coordinates = batch[NOISY_RELATIVE_COORDINATES]
        relative_coordinates_shape = relative_coordinates.shape
        batch_size = relative_coordinates_shape[0]
        assert (
            len(relative_coordinates_shape) == 3 and relative_coordinates_shape[2] == self.spatial_dimension
        ), "The relative coordinates are expected to be in a tensor of shape [batch_size, number_of_atoms, 3]"

        assert torch.logical_and(
            relative_coordinates >= 0.0, relative_coordinates < 1.0
        ).all(), "All components of the relative coordinates are expected to be in [0,1)."

        assert TIME in batch, f"The time step should be present in the batch dictionary with key '{TIME}'"

        times = batch[TIME]
        time_shape = times.shape
        assert (
            time_shape[0] == batch_size
        ), "the batch size dimension is inconsistent between positions and time steps."
        assert (
            len(time_shape) == 2 and time_shape[1] == 1
        ), "The time steps are expected to be in a tensor of shape [batch_size, 1]"

        assert torch.logical_and(
            times >= 0.0, times <= 1.0
        ).all(), "The times are expected to be normalized between 0 and 1."

        assert NOISE in batch, "There should be a 'noise' parameter in the batch dictionary."
        assert batch[NOISE].shape == times.shape, "the 'noise' parameter should have the same shape as the 'time'."

        assert UNIT_CELL in batch, f"The unit cell should be present in the batch dictionary with key '{UNIT_CELL}'"

        unit_cell = batch[UNIT_CELL]
        unit_cell_shape = unit_cell.shape
        assert (
            unit_cell_shape[0] == batch_size
        ), "the batch size dimension is inconsistent between positions and unit cell."
        assert (
            len(unit_cell_shape) == 3 and unit_cell_shape[1] == self.spatial_dimension
            and unit_cell_shape[2] == self.spatial_dimension
        ), "The unit cell is expected to be in a tensor of shape [batch_size, spatial_dimension, spatial_dimension]."
        à
        if self.conditional_prob > 0:
            assert CARTESIAN_FORCES in batch, \
                (f"The cartesian forces should be present in "
                 f"the batch dictionary with key '{CARTESIAN_FORCES}'")

            cartesian_forces = batch[CARTESIAN_FORCES]
            cartesian_forces_shape = cartesian_forces.shape
            assert (
                    len(cartesian_forces_shape) == 3 and cartesian_forces_shape[2] == self.spatial_dimension
            ), "The cartesian forces are expected to be in a tensor of shape [batch_size, number_of_atoms,"
            + f"{self.spatial_dimension}]"

    def forward(self, batch: Dict[AnyStr, torch.Tensor], conditional: Optional[bool] = None) -> torch.Tensor:
        """Model forward.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional: if True, do an conditional forward, if False, do a unconditional forward. If None, choose
                randomly with probability conditional_prob

        Returns:
            computed_scores : the scores computed by the model.
        """
        self._check_batch(batch)
        if conditional is None:
            conditional = torch.rand(1,) < self.conditional_prob
        if conditional:
            return self._forward_unchecked(batch, conditional=False)
        else:
            return (self._forward_unchecked(batch, conditional=True) * self.conditional_gamma
                    + self._forward_unchecked(batch, conditional=False) * (1 - self.conditional_gamma))

    def _forward_unchecked(self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False) -> torch.Tensor:
        """Forward unchecked.

        This method assumes that the input data has already been checked with respect to expectations
        and computes the scores assuming that the data is in the correct format.

        This method should be implemented in the derived class.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional (optional): if True, do a forward as though the model was conditional on the forces.

        Returns:
            computed_scores : the scores computed by the model.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class MLPScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for MLP score networks."""

    architecture: str = 'mlp'
    number_of_atoms: int  # the number of atoms in a configuration.
    n_hidden_dimensions: int  # the number of hidden layers.
    hidden_dimensions_size: int  # the dimensions of the hidden layers.
    condition_embedding_size: int = 64  # dimension of the conditional variable embedding


class MLPScoreNetwork(ScoreNetwork):
    """Simple Model Class.

    Inherits from the given framework's model class. This is a simple MLP model.
    """

    def __init__(self, hyper_params: MLPScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(MLPScoreNetwork, self).__init__(hyper_params)
        hidden_dimensions = [hyper_params.hidden_dimensions_size] * hyper_params.n_hidden_dimensions
        self._natoms = hyper_params.number_of_atoms

        output_dimension = self.spatial_dimension * self._natoms
        input_dimension = output_dimension + 1

        self.condition_embedding_layer = nn.Linear(output_dimension, hyper_params.condition_embedding_size)

        self.flatten = nn.Flatten()
        self.mlp_layers = nn.ModuleList()
        self.conditional_layers = nn.ModuleList()
        input_dimensions = [input_dimension] + hidden_dimensions
        output_dimensions = hidden_dimensions + [output_dimension]

        for input_dimension, output_dimension, add_relu in zip(input_dimensions, output_dimensions):
            self.mlp_layers.append(nn.Linear(input_dimension, output_dimension))
            self.conditional_layers.append(nn.Linear(hyper_params.condition_embedding_size, output_dimension))
        self.non_linearity = nn.ReLU()

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        super(MLPScoreNetwork, self)._check_batch(batch)
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
            computed_scores : the scores computed by the model.
        """
        relative_coordinates = batch[NOISY_RELATIVE_COORDINATES]
        # shape [batch_size, number_of_atoms, spatial_dimension]

        times = batch[TIME].to(relative_coordinates.device)  # shape [batch_size, 1]
        input = torch.cat([self.flatten(relative_coordinates), times], dim=1)

        forces_input = self.condition_embedding_layer(self.flatten(batch[CARTESIAN_FORCES]))

        for i, (layer, condition_layer) in enumerate(zip(self.mlp_layers, self.conditional_layers)):
            if i != 0:
                input = self.non_linearity(input)
            input = layer(input)
            if conditional:
                input += condition_layer(forces_input)

        output = self.mlp_layers(input).reshape(relative_coordinates.shape)
        return output


@dataclass(kw_only=True)
class MACEScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for MACE score networks."""

    architecture: str = 'mace'
    number_of_atoms: int  # the number of atoms in a configuration.
    use_pretrained: Optional[str] = None  # if None, do not use pre-trained ; else, should be in small, medium or large
    pretrained_weights_path: str = './'  # path to pre-trained model weights
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
    gate: str = "silu"  # non linearity for last readout - choices: ["silu", "tanh", "abs", "None"]
    radial_MLP: List[int] = field(default_factory=lambda: [64, 64, 64])  # "width of the radial MLP"
    radial_type: str = "bessel"  # type of radial basis functions - choices=["bessel", "gaussian", "chebyshev"]
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

        self.z_table = get_atomic_number_table_from_zs(list(range(89)))  # need 89 for pre-trained model
        # dataloader
        self.r_max = hyper_params.r_max
        self.collate_fn = Collater(follow_batch=[None], exclude_keys=[None])

        mace_config = dict(
            r_max=hyper_params.r_max,
            num_bessel=hyper_params.num_bessel,
            num_polynomial_cutoff=hyper_params.num_polynomial_cutoff,
            max_ell=hyper_params.max_ell,
            interaction_cls=interaction_classes[hyper_params.interaction_cls],
            interaction_cls_first=interaction_classes[hyper_params.interaction_cls_first],
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
            radial_type=hyper_params.radial_type
        )

        self._natoms = hyper_params.number_of_atoms

        if hyper_params.use_pretrained is None or hyper_params.use_pretrained == 'None':
            self.mace_network = MACE(**mace_config)
            output_node_features_irreps = (
                build_mace_output_nodes_irreducible_representation(hyper_params.hidden_irreps,
                                                                   hyper_params.num_interactions))
        else:
            output_node_features_irreps = get_pretrained_mace_output_node_features_irreps(hyper_params.use_pretrained)
            self.mace_network, mace_output_size = get_pretrained_mace(hyper_params.use_pretrained,
                                                                      hyper_params.pretrained_weights_path)
            assert output_node_features_irreps.dim == mace_output_size, "Something is wrong with pretrained dimensions."

        self.mace_output_size = output_node_features_irreps.dim
        self.prediction_head = instantiate_mace_prediction_head(output_node_features_irreps,
                                                                hyper_params.prediction_head_parameters)

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        super(MACEScoreNetwork, self)._check_batch(batch)
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
        del conditional  # TODO implement conditional
        relative_coordinates = batch[NOISY_RELATIVE_COORDINATES]
        batch[NOISY_CARTESIAN_POSITIONS] = torch.bmm(relative_coordinates, batch[UNIT_CELL])  # positions in Angstrom
        graph_input = input_to_mace(batch, radial_cutoff=self.r_max)
        mace_output = self.mace_network(graph_input, compute_force=False, training=self.training)

        # The node features are organized as (batchsize * natoms, output_size) in the mace output because
        # torch_geometric puts all the graphs in a batch in a single large graph.
        flat_node_features = mace_output['node_feats']

        # The times have a single value per batch element; we repeat the array so that there is one value per atom,
        # with this value the same for all atoms belonging to the same graph.
        times = batch[TIME].to(relative_coordinates.device)  # shape [batch_size, 1]
        flat_times = times[graph_input.batch]  # shape [batch_size * natoms, 1]
        flat_scores = self.prediction_head(flat_node_features, flat_times)  # shape [batch_size * natoms, spatial_dim]

        # Reshape the scores to have an explicit batch dimension
        scores = flat_scores.reshape(-1, self._natoms, self.spatial_dimension)

        return scores


@dataclass(kw_only=True)
class DiffusionMACEScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for Diffusion MACE score networks."""
    architecture: str = 'diffusion_mace'
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

        diffusion_mace_output = self.diffusion_mace_network(graph_input, compute_force=True, training=self.training)

        # Diffusion MACE operates in Euclidean space. The computed "forces" are the negative gradient of the "energy"
        flat_cartesian_scores = -diffusion_mace_output['forces']
        cartesian_scores = flat_cartesian_scores.reshape(batch_size, number_of_atoms, spatial_dimension)

        # Using the chain rule, we can derive nabla_x given nabla_r, where 'x' is relative coordinates and 'r'
        # is cartesian space.
        scores = torch.bmm(cartesian_scores, basis_vectors.transpose(2, 1))

        return scores

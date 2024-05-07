"""Score Network.

This module implements score networks for positions in relative coordinates.
Relative coordinates are with respect to lattice vectors which define the
periodic unit cell.
"""
import os
from dataclasses import dataclass, field
from typing import AnyStr, Dict, List

import numpy as np
import torch
from e3nn import o3  # e3nn is packaged with mace-torch
from mace.modules import gate_dict, interaction_classes
from mace.modules.models import MACE
from mace.tools import get_atomic_number_table_from_zs
from mace.tools.torch_geometric.dataloader import Collater
from torch import nn

from crystal_diffusion.models.mace_utils import input_to_mace

# mac fun time
# for mace, conflict with mac
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already- \
# initial
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@dataclass(kw_only=True)
class ScoreNetworkParameters:
    """Base Hyper-parameters for score networks."""

    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live.


class ScoreNetwork(torch.nn.Module):
    """Base score network.

    This base class defines the interface that all score networks should have
    in order to be easily interchangeable (ie, polymorphic).
    """

    position_key = "noisy_relative_positions"  # unitless positions in the lattice coordinate basis
    timestep_key = "time"
    unit_cell_key = "unit_cell"  # unit cell definition in Angstrom

    def __init__(self, hyper_params: ScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyperparameters from the config file.
        """
        super(ScoreNetwork, self).__init__()
        self._hyper_params = hyper_params
        self.spatial_dimension = hyper_params.spatial_dimension

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        """Check batch.

        Check that the batch dictionary contains the expected inputs, and that
        those inputs have the expected dimensions.

        It is expected that:
            - the positions are present and of shape [batch_size, number of atoms, spatial_dimension]
            - all the components of positions  will be in [0, 1)
            - the time steps are present and of shape [batch_size, 1]
            - the time steps are in range [0, 1].

        An assert will fail if the batch does not conform with expectation.

        Args:
            batch : dictionary containing the data to be processed by the model.

        Returns:
            None.
        """
        assert (
            self.position_key in batch
        ), f"The positions should be present in the batch dictionary with key '{self.position_key}'"

        positions = batch[self.position_key]
        position_shape = positions.shape
        batch_size = position_shape[0]
        assert (
            len(position_shape) == 3 and position_shape[2] == self.spatial_dimension
        ), "The positions are expected to be in a tensor of shape [batch_size, number_of_atoms, 3]"

        assert torch.logical_and(
            positions >= 0.0, positions < 1.0
        ).all(), "All components of the positions are expected to be in [0,1)."

        assert (
            self.timestep_key in batch
        ), f"The time step should be present in the batch dictionary with key '{self.timestep_key}'"

        times = batch[self.timestep_key]
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

        assert (
            self.unit_cell_key in batch
        ), f"The unit cell should be present in the batch dictionary with key '{self.unit_cell_key}'"

        unit_cell = batch[self.unit_cell_key]
        unit_cell_shape = unit_cell.shape
        assert (
            unit_cell_shape[0] == batch_size
        ), "the batch size dimension is inconsistent between positions and unit cell."
        assert (
            len(unit_cell_shape) == 3 and unit_cell_shape[1] == self.spatial_dimension
            and unit_cell_shape[2] == self.spatial_dimension
        ), "The unit cell is expected to be in a tensor of shape [batch_size, spatial_dimension, spatial_dimension]."

    def forward(self, batch: Dict[AnyStr, torch.Tensor]) -> torch.Tensor:
        """Model forward.

        Args:
            batch : dictionary containing the data to be processed by the model.

        Returns:
            computed_scores : the scores computed by the model.
        """
        self._check_batch(batch)
        return self._forward_unchecked(batch)

    def _forward_unchecked(self, batch: Dict[AnyStr, torch.Tensor]) -> torch.Tensor:
        """Forward unchecked.

        This method assumes that the input data has already been checked with respect to expectations
        and computes the scores assuming that the data is in the correct format.

        This method should be implemented in the derived class.

        Args:
            batch : dictionary containing the data to be processed by the model.

        Returns:
            computed_scores : the scores computed by the model.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class MLPScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for MLP score networks."""

    number_of_atoms: int  # the number of atoms in a configuration.
    n_hidden_dimensions: int  # the number of hidden layers.
    hidden_dimensions_size: int  # the dimensions of the hidden layers.


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

        self.flatten = nn.Flatten()
        self.mlp_layers = nn.Sequential()
        input_dimensions = [input_dimension] + hidden_dimensions
        output_dimensions = hidden_dimensions + [output_dimension]
        add_relus = len(input_dimensions) * [True]
        add_relus[-1] = False

        for input_dimension, output_dimension, add_relu in zip(input_dimensions, output_dimensions, add_relus):
            self.mlp_layers.append(nn.Linear(input_dimension, output_dimension))
            if add_relu:
                self.mlp_layers.append(nn.ReLU())

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        super(MLPScoreNetwork, self)._check_batch(batch)
        number_of_atoms = batch[self.position_key].shape[1]
        assert (
            number_of_atoms == self._natoms
        ), "The dimension corresponding to the number of atoms is not consistent with the configuration."

    def _forward_unchecked(self, batch: Dict[AnyStr, torch.Tensor]) -> torch.Tensor:
        """Forward unchecked.

        This method assumes that the input data has already been checked with respect to expectations
        and computes the scores assuming that the data is in the correct format.

        Args:
            batch : dictionary containing the data to be processed by the model.

        Returns:
            computed_scores : the scores computed by the model.
        """
        positions = batch[self.position_key]  # shape [batch_size, number_of_atoms, spatial_dimension]
        times = batch[self.timestep_key].to(positions.device)  # shape [batch_size, 1]
        input = torch.cat([self.flatten(positions), times], dim=1)

        output = self.mlp_layers(input).reshape(positions.shape)
        return output


@dataclass(kw_only=True)
class MACEScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for MACE score networks."""

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
    atomic_energies: np.ndarray = field(default_factory=lambda: np.zeros((1,)))
    avg_num_neighbors: int = 1  # normalization factor for the message
    correlation: int = 3
    gate: str = "silu"  # non linearity for last readout - choices: ["silu", "tanh", "abs", "None"]
    radial_MLP: List[int] = field(default_factory=lambda: [64, 64, 64])  # "width of the radial MLP"
    radial_type: str = "bessel"  # type of radial basis functions - choices=["bessel", "gaussian", "chebyshev"]
    n_hidden_dimensions: int  # the number of hidden layers for the final MLP - should be small
    hidden_dimensions_size: int  # the dimensions of the hidden layers - should be small


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

        self.z_table = get_atomic_number_table_from_zs([14])  # TODO more than 1 type & duplicate from dataloader
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

        self.mace_network = MACE(**mace_config)

        hidden_dimensions = [hyper_params.hidden_dimensions_size] * hyper_params.n_hidden_dimensions
        hidden_irreps_size = int(hyper_params.hidden_irreps.split('x')[0])  # needs to be the same for 0e as for 1o
        hidden_irreps_dim = 1 + 3 * ('1o' in hyper_params.hidden_irreps)
        self.mace_output_size = ((hyper_params.num_interactions - 1) * hidden_irreps_dim + 1) * hidden_irreps_size
        # mace_output_size is 640 by default: ((2 - 1) * 4 + 1) * 128 = 5 * 128 = 640
        self.mlp_layers = nn.Sequential()
        # TODO we could add a linear layer to the times before concat with mace_output
        input_dimensions = [self.mace_output_size + 1] + hidden_dimensions  # add 1 for the times
        output_dimensions = hidden_dimensions + [self.spatial_dimension]
        add_relus = len(input_dimensions) * [True]
        add_relus[-1] = False

        for input_dimension, output_dimension, add_relu in zip(input_dimensions, output_dimensions, add_relus):
            self.mlp_layers.append(nn.Linear(input_dimension, output_dimension))
            if add_relu:
                self.mlp_layers.append(nn.ReLU())

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        super(MACEScoreNetwork, self)._check_batch(batch)
        number_of_atoms = batch[self.position_key].shape[1]
        assert (
            number_of_atoms == self._natoms
        ), "The dimension corresponding to the number of atoms is not consistent with the configuration."

    def _forward_unchecked(self, batch: Dict[AnyStr, torch.Tensor]) -> torch.Tensor:
        """Forward unchecked.

        This method assumes that the input data has already been checked with respect to expectations
        and computes the scores assuming that the data is in the correct format.

        Args:
            batch : dictionary containing the data to be processed by the model.

        Returns:
            output : the scores computed by the model as a [batch_size, n_atom, spatial_dimension] tensor.
        """
        positions = batch[self.position_key]
        batch['abs_positions'] = torch.bmm(positions, batch[self.unit_cell_key])  # positions in Angstrom
        graph_input = input_to_mace(batch, self.unit_cell_key, radial_cutoff=self.r_max)
        mace_output = self.mace_network(graph_input, compute_force=False, training=self.training)
        # we want the node features but they are organized as (batchsize * natoms, output_size) because
        # torch_geometric puts all the graphs in a batch in a single large graph
        mace_output = mace_output['node_feats'].view(-1, self._natoms, self.mace_output_size)
        times = batch[self.timestep_key].to(positions.device)  # shape [batch_size, 1]
        times = times.unsqueeze(1).repeat(1, self._natoms, 1)  # shape [batch_size, natoms, 1]
        mlp_input = torch.cat([mace_output, times], dim=-1)
        # pass through the final MLP layers
        output = self.mlp_layers(mlp_input)
        return output

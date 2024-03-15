"""Score Network.

This module implements score networks for positions in relative coordinates.
Relative coordinates are with respect to lattice vectors which define the
periodic unit cell.
"""
from dataclasses import dataclass
from typing import AnyStr, Dict

import torch
from torch import nn


@dataclass(kw_only=True)
class BaseScoreNetworkParameters:
    """Base Hyper-parameters for score networks."""
    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live.


class BaseScoreNetwork(torch.nn.Module):
    """Base score network.

    This base class defines the interface that all score networks should have
    in order to be easily interchangeable (ie, polymorphic).
    """
    position_key = "noisy_relative_positions"  # unitless positions in the lattice coordinate basis
    timestep_key = "time"

    def __init__(self, hyper_params: BaseScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyperparameters from the config file.
        """
        super(BaseScoreNetwork, self).__init__()
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
class MLPScoreNetworkParameters(BaseScoreNetworkParameters):
    """Specific Hyper-parameters for MLP score networks."""
    number_of_atoms: int  # the number of atoms in a configuration.
    hidden_dim: int


class MLPScoreNetwork(BaseScoreNetwork):
    """Simple Model Class.

    Inherits from the given framework's model class. This is a simple MLP model.
    """

    def __init__(self, hyper_params: MLPScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params (dict): hyper parameters from the config file.
        """
        super(MLPScoreNetwork, self).__init__(hyper_params)
        hidden_dim = hyper_params.hidden_dim
        self._natoms = hyper_params.number_of_atoms

        output_dimension = self.spatial_dimension * self._natoms
        input_dimension = output_dimension + 1

        self.flatten = nn.Flatten()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dimension, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dimension),
        )

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
        times = batch[self.timestep_key]  # shape [batch_size, 1]

        input = torch.cat([self.flatten(positions), times], dim=1)

        output = self.mlp_layers(input).reshape(positions.shape)
        return output

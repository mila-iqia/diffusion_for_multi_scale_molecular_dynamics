"""Score Network.

This module implements score networks for positions in relative coordinates.
Relative coordinates are with respect to lattice vectors which define the
periodic unit cell.
"""
import os
from dataclasses import dataclass
from typing import AnyStr, Dict, Optional

import torch

from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)

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
    conditional_prob: float = 0.  # probability of making a conditional forward - else, do a unconditional forward
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

        if self.conditional_prob > 0:
            assert CARTESIAN_FORCES in batch, \
                (f"The cartesian forces should be present in "
                 f"the batch dictionary with key '{CARTESIAN_FORCES}'")

            cartesian_forces = batch[CARTESIAN_FORCES]
            cartesian_forces_shape = cartesian_forces.shape
            assert (
                len(cartesian_forces_shape) == 3 and cartesian_forces_shape[2] == self.spatial_dimension
            ), ("The cartesian forces are expected to be in a tensor of shape [batch_size, number_of_atoms,"
                f"{self.spatial_dimension}]")

    def forward(self, batch: Dict[AnyStr, torch.Tensor], conditional: Optional[bool] = None) -> torch.Tensor:
        """Model forward.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional: if True, do a conditional forward, if False, do a unconditional forward. If None, choose
                randomly with probability conditional_prob

        Returns:
            computed_scores : the scores computed by the model.
        """
        self._check_batch(batch)
        if conditional is None:
            conditional = torch.rand(1,) < self.conditional_prob
        if not conditional:
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

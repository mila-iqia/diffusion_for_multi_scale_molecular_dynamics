r"""Score Network.

This module implements score networks for positions in relative coordinates, atom types diffusion and lattice parameters
diffusion. Relative coordinates are with respect to lattice vectors which define the
periodic unit cell.

The coordinates part of the output aims to calculate

.. math::
    output.X \propto nabla_X \ln P(x,t)

where X is relative coordinates.
"""

from dataclasses import dataclass
from typing import AnyStr, Dict, Optional

import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_number_of_lattice_parameters


@dataclass(kw_only=True)
class ScoreNetworkParameters:
    """Base Hyper-parameters for score networks."""

    architecture: str
    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live.
    num_atom_types: int  # number of possible atomic species - not counting the MASK class used in the diffusion
    conditional_prob: float = (
        0.0  # probability of making a conditional forward - else, do an unconditional forward
    )
    conditional_gamma: float = (
        2.0  # conditional score weighting - see eq. B45 in MatterGen
    )
    # p_\gamma(x|c) = p(c|x)^\gamma p(x)

    def __post_init__(self):
        self.num_lattice_parameters = get_number_of_lattice_parameters(
            self.spatial_dimension
        )


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
        self.num_atom_types = hyper_params.num_atom_types
        self.conditional_prob = hyper_params.conditional_prob
        self.conditional_gamma = hyper_params.conditional_gamma

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        """Check batch.

        Check that the batch dictionary contains the expected inputs, and that
        those inputs have the expected dimensions.

        It is expected that:
            - an AXL namedtuple is present with
              - the relative coordinates of shape [batch_size, number of atoms, spatial_dimension]
              - the atom types of shape [batch_size, number of atoms]
              - the unit cell vectors  TODO shape
            - all the components of relative coordinates will be in [0, 1)
            - all the components of atom types are integers between [0, number of atomic species + 1)
                the + 1 accounts for the MASK class
            - the time steps are present and of shape [batch_size, 1]
            - the time steps are in range [0, 1].
            - the 'noise' parameter sigma is present and has the same shape as time.

        An assert will fail if the batch does not conform with expectation.

        Args:
            batch : dictionary containing the data to be processed by the model.

        Returns:
            None.
        """
        assert NOISY_AXL_COMPOSITION in batch, (
            f"The noisy coordinates, atomic types and lattice vectors should be present in "
            f"the batch dictionary with key '{NOISY_AXL_COMPOSITION}'"
        )

        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        relative_coordinates_shape = relative_coordinates.shape
        batch_size = relative_coordinates_shape[0]
        assert (
            len(relative_coordinates_shape) == 3
            and relative_coordinates_shape[2] == self.spatial_dimension
        ), (
            "The relative coordinates are expected to be in a tensor of "
            "shape [batch_size, number_of_atoms, spatial_dimension]"
        )

        assert torch.logical_and(
            relative_coordinates >= 0.0, relative_coordinates < 1.0
        ).all(), (
            "All components of the relative coordinates are expected to be in [0,1)."
        )

        assert (
            TIME in batch
        ), f"The time step should be present in the batch dictionary with key '{TIME}'"

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

        assert (
            NOISE in batch
        ), "There should be a 'noise' parameter in the batch dictionary."
        assert (
            batch[NOISE].shape == times.shape
        ), "the 'noise' parameter should have the same shape as the 'time'."

        lattice_parameters = batch[NOISY_AXL_COMPOSITION].L
        lattice_parameters_shape = lattice_parameters.shape
        assert (
            lattice_parameters_shape[0] == batch_size
        ), "the batch size dimension is inconsistent between positions and unit cell."
        assert len(lattice_parameters_shape) == 2 and lattice_parameters_shape[
            1
        ] == int(self.spatial_dimension * (self.spatial_dimension + 1) / 2), (
            "The lattice parameters are expected to be in a tensor of shape [batch_size, spatial_dimension * "
            "(spatial_dimension + 1) / 2].}"
        )

        atom_types = batch[NOISY_AXL_COMPOSITION].A
        atom_types_shape = atom_types.shape
        assert (
            atom_types_shape[0] == batch_size
        ), "the batch size dimension is inconsistent between positions and atom types."
        assert (
            len(atom_types_shape) == 2
        ), "The atoms type are expected to be in a tensor of shape [batch_size, number of atoms]."

        assert torch.logical_and(
            atom_types >= 0,
            atom_types
            < self.num_atom_types + 1,  # MASK is a possible type in a noised sample
        ).all(), f"All atom types are expected to be in [0, {self.num_atom_types}]."

        if self.conditional_prob > 0:
            assert CARTESIAN_FORCES in batch, (
                f"The cartesian forces should be present in "
                f"the batch dictionary with key '{CARTESIAN_FORCES}'"
            )

            cartesian_forces = batch[CARTESIAN_FORCES]
            cartesian_forces_shape = cartesian_forces.shape
            assert (
                len(cartesian_forces_shape) == 3
                and cartesian_forces_shape[2] == self.spatial_dimension
            ), (
                "The cartesian forces are expected to be in a tensor of shape [batch_size, number_of_atoms,"
                f"{self.spatial_dimension}]"
            )

    def _impose_non_mask_atomic_type_prediction(self, output: AXL):
        # Force the last logit to be -infinity, making it impossible for the model to predict MASK.
        output.A[..., self.num_atom_types] = -torch.inf

    def forward(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: Optional[bool] = None
    ) -> AXL:
        """Model forward.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional: if True, do a conditional forward, if False, do a unconditional forward. If None, choose
                randomly with probability conditional_prob

        Returns:
            computed_scores : the scores computed by the model in an AXL namedtuple.
        """
        self._check_batch(batch)
        if conditional is None:
            conditional = (
                torch.rand(
                    1,
                )
                < self.conditional_prob
            )

        if not conditional:
            output = self._forward_unchecked(batch, conditional=False)
        else:
            # TODO this is not going to work
            output = self._forward_unchecked(
                batch, conditional=True
            ) * self.conditional_gamma + self._forward_unchecked(
                batch, conditional=False
            ) * (
                1 - self.conditional_gamma
            )

        self._impose_non_mask_atomic_type_prediction(output)

        return output

    def _forward_unchecked(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> AXL:
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

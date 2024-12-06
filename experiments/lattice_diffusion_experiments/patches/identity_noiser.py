import logging

import torch

from diffusion_for_multi_scale_molecular_dynamics.noisers.atom_types_noiser import \
    AtomTypesNoiser
from diffusion_for_multi_scale_molecular_dynamics.noisers.relative_coordinates_noiser import \
    RelativeCoordinatesNoiser

logger = logging.getLogger(__name__)


class RelativeCoordinatesIdentityNoiser(RelativeCoordinatesNoiser):
    """Identity Noiser for the relative coordinates.

    This class can be used as a stand-in that returns the identity (ie, no noising).
    """

    @staticmethod
    def get_noisy_relative_coordinates_sample(
        real_relative_coordinates: torch.Tensor, sigmas: torch.Tensor
    ) -> torch.Tensor:
        """Get noisy relative coordinates sample."""
        logger.debug("Identity Noiser! Return input as output.")
        return real_relative_coordinates


class AtomTypesIdentityNoiser(AtomTypesNoiser):
    """Identity Noiser for the atom types.

    This class can be used as a stand-in that returns the identity (ie, no noising).
    """

    @staticmethod
    def get_noisy_atom_types_sample(
        real_onehot_atom_types: torch.Tensor, q_bar: torch.Tensor
    ) -> torch.Tensor:
        """Get noisy atom types sample."""
        logger.debug("Identity Noiser! Return input as output.")
        return torch.argmax(real_onehot_atom_types, dim=-1)

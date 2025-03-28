import logging

import torch

from diffusion_for_multi_scale_molecular_dynamics.noisers.relative_coordinates_noiser import \
    RelativeCoordinatesNoiser

logger = logging.getLogger(__name__)


class IdentityRelativeCoordinatesNoiser(RelativeCoordinatesNoiser):
    """Identity Relative Coordinates Noiser.

    This class can be used as a stand-in that returns the identity (ie, no noising). This
    is useful for sanity checking diffusion on atom-types only.
    """

    @staticmethod
    def get_noisy_relative_coordinates_sample(
        real_relative_coordinates: torch.Tensor, sigmas: torch.Tensor
    ) -> torch.Tensor:
        """Get noisy relative coordinates sample."""
        logger.debug("Identity Noiser! Return input as output.")
        return real_relative_coordinates

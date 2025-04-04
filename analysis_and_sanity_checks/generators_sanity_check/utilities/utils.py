from typing import Any, AnyStr, Dict

import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.sample_trajectory import \
    SampleTrajectory


class DisplacementCalculator:
    """Calculate the displacement distribution."""

    def __init__(self, equilibrium_relative_coordinates: torch.Tensor):
        """Init method."""
        self.equilibrium_relative_coordinates = equilibrium_relative_coordinates

    def compute_displacements(
        self, batch_relative_coordinates: torch.Tensor

    ) -> np.ndarray:
        """Compute displacements."""
        return (batch_relative_coordinates - self.equilibrium_relative_coordinates).flatten().numpy()


def standardize_sde_trajectory_data(sde_sample_trajectory: SampleTrajectory) -> Dict[AnyStr, Any]:
    """Utility method to extract relevant data from the internal data of the SampleTrajectory object."""
    raw_data = sde_sample_trajectory._internal_data['sde'][0]

    times = raw_data['times']
    relative_coordinates = raw_data['relative_coordinates']

    return dict(time=times, relative_coordinates=relative_coordinates)

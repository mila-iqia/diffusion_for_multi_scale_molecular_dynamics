from typing import Callable

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    CARTESIAN_FORCES, NOISE, NOISY_RELATIVE_COORDINATES, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler


def get_normalized_score_function(
    noise_parameters: NoiseParameters,
    sigma_normalized_score_network: ScoreNetwork,
    basis_vectors: torch.Tensor,
) -> Callable:
    """Get normalizd score function."""
    variance_calculator = NoiseScheduler(noise_parameters)

    def normalized_score_function(
        relative_coordinates: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        batch_size, number_of_atoms, spatial_dimension = relative_coordinates.shape
        unit_cells = einops.repeat(
            basis_vectors.to(relative_coordinates), "s1 s2 -> b s1 s2", b=batch_size
        )

        forces = torch.zeros_like(relative_coordinates)
        sigmas = variance_calculator.get_sigma(times)

        augmented_batch = {
            NOISY_RELATIVE_COORDINATES: relative_coordinates,
            TIME: times,
            NOISE: sigmas,
            UNIT_CELL: unit_cells,
            CARTESIAN_FORCES: forces,
        }

        sigma_normalized_scores = sigma_normalized_score_network(
            augmented_batch, conditional=False
        )

        return sigma_normalized_scores

    return normalized_score_function

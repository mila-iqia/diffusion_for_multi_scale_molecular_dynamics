from typing import Callable

import einops
import numpy as np
import torch
from torch.func import jacrev

from crystal_diffusion.models.score_networks import ScoreNetwork
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)
from crystal_diffusion.samplers.exploding_variance import ExplodingVariance
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell


def create_fixed_time_vector_field_function(
    sigma_normalized_score_network: ScoreNetwork,
    noise_parameters: NoiseParameters,
    time: float,
    basis_vectors: torch.Tensor,
):
    """Create the vector field function."""
    variance_calculator = ExplodingVariance(noise_parameters)

    def vector_field_function(relative_coordinates: torch.Tensor) -> torch.Tensor:
        batch_size, number_of_atoms, spatial_dimension = relative_coordinates.shape

        times = einops.repeat(
            torch.tensor([time]).to(relative_coordinates), "1 -> b 1", b=batch_size
        )
        unit_cells = einops.repeat(
            basis_vectors.to(relative_coordinates), "s1 s2 -> b s1 s2", b=batch_size
        )

        forces = torch.zeros_like(relative_coordinates)
        sigmas = variance_calculator.get_sigma(times)
        g2 = variance_calculator.get_g_squared(times)

        vector_field_prefactor = -g2 / sigmas

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

        prefactor = einops.repeat(
            vector_field_prefactor,
            "b 1 -> b n s",
            n=number_of_atoms,
            s=spatial_dimension,
        )

        return prefactor * sigma_normalized_scores

    return vector_field_function


def get_hessian_function(vector_field_function: Callable) -> Callable:
    """Get hessian function."""
    batch_hessian_function = jacrev(vector_field_function, argnums=0)

    def hessian_function(relative_coordinates: torch.Tensor) -> torch.Tensor:
        # The batch hessian has dimension [batch_size, natoms, space, batch_size, natoms, space]
        batch_hessian = batch_hessian_function(relative_coordinates)

        # Diagonal dumps the batch dimension at the end.
        hessian = torch.diagonal(batch_hessian, dim1=0, dim2=3)

        flat_hessian = einops.rearrange(hessian, "n1 s1 n2 s2 b -> b (n1 s1) (n2 s2)")
        return flat_hessian

    return hessian_function


def get_flat_vector_field_function(
    vector_field_function: Callable, number_of_atoms: int, spatial_dimension: int
) -> Callable:
    """Get a flat vector field function."""
    def flat_vector_field_function(x: np.ndarray) -> np.ndarray:
        cast_x = torch.from_numpy(x).to(torch.float32)
        relative_coordinates = einops.rearrange(
            cast_x, "(n s) -> 1 n s", n=number_of_atoms, s=spatial_dimension
        )
        relative_coordinates = map_relative_coordinates_to_unit_cell(
            relative_coordinates
        )
        vector_field = vector_field_function(relative_coordinates)
        return einops.rearrange(vector_field, "1 n s -> (n s)").numpy()

    return flat_vector_field_function

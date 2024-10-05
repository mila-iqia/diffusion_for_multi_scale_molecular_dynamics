import itertools
from typing import Callable, Tuple

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

def get_normalized_score_function( noise_parameters: NoiseParameters,
                                   sigma_normalized_score_network: ScoreNetwork,
                                   basis_vectors: torch.Tensor)-> Callable:

    variance_calculator = ExplodingVariance(noise_parameters)

    def normalized_score_function(relative_coordinates: torch.Tensor, times: torch.Tensor) -> torch.Tensor:

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

def get_cubic_point_group_symmetries():
    permutations = [torch.diag(torch.ones(3))[[idx]] for idx in itertools.permutations([0, 1, 2])]
    sign_changes = [torch.diag(torch.tensor(diag)) for diag in itertools.product([-1., 1.], repeat=3)]
    symmetries = []
    for permutation in permutations:
        for sign_change in sign_changes:
            symmetries.append(permutation @ sign_change)

    return symmetries
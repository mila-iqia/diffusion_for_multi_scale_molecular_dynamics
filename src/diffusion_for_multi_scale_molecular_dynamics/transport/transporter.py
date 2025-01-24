from typing import Tuple

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.transport.distance import (
    get_geodesic_displacements, get_squared_geodesic_distance)
from diffusion_for_multi_scale_molecular_dynamics.transport.optimal_permutation import \
    get_optimal_permutation
from diffusion_for_multi_scale_molecular_dynamics.transport.optimal_translation import \
    find_squared_geodesic_distance_minimizing_translation
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell


class Transporter:
    """Transporter.

    This class applies an iterative algorithm to find the best symmetry group operation that aligns
    two points on the hyper-torus.

    The symmetry group is composed of translations, point group symmetries and permutations.

    """

    def __init__(
        self,
        point_group_operations: torch.Tensor,
        maximum_number_of_steps: int = 3,
        progress_threshold: float = 1e-8,
    ):
        """Init method.

        Args:
            point_group_operations: matrices representing the point group operations, of dimension
                [number_of_point_group_operations, spatial_dimension, spatial_dimension]
            maximum_number_of_steps: the maximum number of steps that can be performed using the iterative algorithm.
            progress_threshold: the threshold for determining if the algorithm should stop.
        """
        self.point_group_operations = point_group_operations
        self.number_of_point_group_operations = len(self.point_group_operations)

        self.maximum_number_of_steps = maximum_number_of_steps
        self.progress_threshold = progress_threshold

    def get_optimal_transport(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, list, bool]:
        """Get optimal transport.

        This method finds the approximate best permutation, point group operation and translation such that
            D2(x, g.y + tau)
        is as small as possible.

        Args:
            x: a point on the hyper-torus. Assumed to be of dimension [number_of_atoms, spatial_dimension]
            y: a point on the hyper-torus. Assumed to be of dimension [number_of_atoms, spatial_dimension]

        Returns:
            final_y: the aligned value of y
            list_squared_distances: the squared distances obtained during algorithm execution
            algo_converged: this the algorithm converge.
        """
        list_squared_distances = []
        previous_squared_distance = get_squared_geodesic_distance(x, y)
        list_squared_distances.append(previous_squared_distance)

        working_y = y.clone()

        batch_x = einops.repeat(
            x, "n d -> b n d", b=self.number_of_point_group_operations
        )

        list_permutations = []
        list_point_group_operation = []
        list_translations = []

        progress = True
        i_step = 0

        while i_step < self.maximum_number_of_steps and progress:
            i_step += 1
            # Permutation
            optimal_permutation = get_optimal_permutation(x, working_y).to(x.device)
            list_permutations.append(optimal_permutation)
            working_y = torch.matmul(optimal_permutation, working_y)

            # point group operation
            batch_working_y = einops.einsum(
                self.point_group_operations, working_y, "o i j, n j -> o n i"
            )

            # all translations
            batch_tau = find_squared_geodesic_distance_minimizing_translation(
                batch_x, batch_working_y
            )
            batch_working_y = map_relative_coordinates_to_unit_cell(
                batch_working_y + batch_tau.unsqueeze(1)
            )

            batch_squared_distances = (
                get_geodesic_displacements(batch_x, batch_working_y) ** 2
            ).sum(dim=[1, 2])

            # find point group operation + translation that minimizes squared distance
            min_idx = batch_squared_distances.argmin()
            tau = batch_tau[min_idx]
            list_translations.append(tau)

            list_point_group_operation.append(self.point_group_operations[min_idx])

            working_y = batch_working_y[min_idx]
            squared_distance = batch_squared_distances[min_idx]

            list_squared_distances.append(squared_distance)

            progress = (
                previous_squared_distance - squared_distance > self.progress_threshold
            )
            previous_squared_distance = squared_distance

        list_squared_distances = torch.tensor(list_squared_distances)
        algo_converged = not progress

        return working_y, list_squared_distances, algo_converged

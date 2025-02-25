from typing import Tuple

import einops
import torch
from scipy.optimize import linear_sum_assignment

from diffusion_for_multi_scale_molecular_dynamics.transport.distance import \
    get_geodesic_displacements
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell


class Transporter:
    """Transporter.

    This class finds a symmetry group operation that aligns two points on the hyper-torus.
    It does not seek to minimize the distance between the first point and the aligned second point:
    it aims to create a fully equivariant function.

    The symmetry group is composed of translations, point group symmetries and permutations.
    """

    def __init__(
        self,
        point_group_operations: torch.Tensor,
    ):
        """Init method.

        Args:
            point_group_operations: matrices representing the point group operations, of dimension
                [number_of_point_group_operations, spatial_dimension, spatial_dimension]
        """
        self.point_group_operations = point_group_operations
        self.number_of_point_group_operations = len(self.point_group_operations)

    @staticmethod
    def get_atan2_translation(x: torch.Tensor) -> torch.Tensor:
        """Get atan2 translation."""
        two_pi = 2 * torch.pi
        x_bar = torch.cos(two_pi * x).mean(dim=1)
        y_bar = torch.sin(two_pi * x).mean(dim=1)
        return torch.atan2(y_bar, x_bar) / two_pi

    def get_translation_invariant(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the center of mass from a point on the hyper-torus."""
        natoms = x.shape[1]
        x_com = einops.repeat(self.get_atan2_translation(x), 'b d -> b n d', n=natoms)
        return map_relative_coordinates_to_unit_cell(x - x_com)

    def _get_all_cost_matrices(
        self, x_minus_x_com: torch.Tensor, mu_minus_mu_com: torch.Tensor
    ) -> torch.Tensor:
        """Get all cost matrices."""
        natoms = x_minus_x_com.shape[1]

        point_group_mu = einops.einsum(
            self.point_group_operations, mu_minus_mu_com, "o d1 d2, b n d2 -> b o n d1"
        )

        point_group_x = einops.repeat(
            x_minus_x_com, "b n d -> b o n d", o=self.number_of_point_group_operations
        )

        # We have to add a dimension to these arrays because the cost matrix is squared in "number of atoms"

        array_x = einops.repeat(point_group_x, "b o n1 d -> b o n1 n2 d", n2=natoms)
        array_mu = einops.repeat(point_group_mu, "b o n2 d -> b o n1 n2 d", n1=natoms)

        squared_displacements = get_geodesic_displacements(array_x, array_mu) ** 2
        # Sum over spatial dimension and to get the cost matrices, of dimension [b o n1 n2]
        cost_matrices = squared_displacements.sum(dim=-1)
        return cost_matrices

    def _solve_linear_assigment_problem(
        self, computed_cost_matrices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve the linear assignment problem."""
        device = computed_cost_matrices.device
        batch_size = computed_cost_matrices.shape[0]
        cpu = torch.device("cpu")
        list_permutations = []
        list_costs = []

        for cost_matrix in einops.rearrange(
            computed_cost_matrices, " b o n1 n2 -> (b o) n1 n2"
        ).to(cpu):
            # Unfortunately, the LAP problem must be solved on CPU and cannot be batched.
            permutation, cost = self._find_permutation_and_cost(cost_matrix)
            list_permutations.append(permutation)
            list_costs.append(cost)

        permutations = einops.rearrange(
            torch.stack(list_permutations, dim=0),
            "(b o) n1 n2 -> b o n1 n2",
            b=batch_size,
        )

        costs = einops.rearrange(
            torch.stack(list_costs, dim=0), "(b o) -> b o", b=batch_size
        )

        lowest_cost_symmetry_operation_indices = costs.argmin(dim=1)

        lowest_cost_permutations = permutations[
            range(batch_size), lowest_cost_symmetry_operation_indices
        ].to(device)
        lowest_cost_point_group_operations = self.point_group_operations[
            lowest_cost_symmetry_operation_indices
        ]
        return lowest_cost_permutations, lowest_cost_point_group_operations

    def _find_permutation_and_cost(self, cost_matrix: torch.Tensor):
        """Find permutation and cost.

        Args:
            cost_matrix: an n x n cost matrix. The tensor is assumed to be on the CPU.

        Returns:
            permutation: the optimal permutation matrix that solves the assignment problem, meaning
                        Tr[permutation . cost_matrix] is minimized.
            cost: the minimized cost.
        """
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        cost = cost_matrix[row_idx, col_idx].sum()
        n = cost_matrix.shape[0]
        optimal_permutation = torch.eye(n)[:, col_idx]
        return optimal_permutation, cost

    def get_optimal_transport(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Get optimal transport.

        This method finds the best symmetry group image of mu to insure that the selection is equivariant
        under symetry operations on x. It is chosen so that the distance is as small as possible, but
        it is not guaranteed to be minimized.

        Args:
            x: a point on the hyper-torus. Assumed to be of dimension [batch_size, number_of_atoms, spatial_dimension]
            mu: a point on the hyper-torus. Assumed to be of dimension [batch_size, number_of_atoms, spatial_dimension]

        Returns:
            aligned_mu: the aligned value of mu
        """
        x_minus_x_com = self.get_translation_invariant(x)
        mu_minus_mu_com = self.get_translation_invariant(mu)

        # Dimension [batch_size, number of point group operations, natoms, natoms]
        computed_cost_matrices = self._get_all_cost_matrices(
            x_minus_x_com, mu_minus_mu_com
        )

        # One permutation and one point group operation per batch entry.
        lowest_cost_permutations, lowest_cost_point_group_operations = (
            self._solve_linear_assigment_problem(computed_cost_matrices)
        )

        # Build the best aligned image of mu
        rotation = einops.einsum(
            lowest_cost_point_group_operations,
            mu_minus_mu_com,
            "b d1 d2, b n d2 -> b n d1",
        )

        # Careful! We must apply the inverse of the permutation, which is its transpose
        rotation_permutation = einops.einsum(
            lowest_cost_permutations, rotation, "b n2 n1, b n2 d -> b n1 d"
        )

        best_mu_image = map_relative_coordinates_to_unit_cell(rotation_permutation)

        return best_mu_image

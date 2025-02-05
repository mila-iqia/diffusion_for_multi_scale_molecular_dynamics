import torch
from scipy.optimize import linear_sum_assignment

from diffusion_for_multi_scale_molecular_dynamics.transport.distance import \
    get_squared_geodesic_distance_cost_matrix


def get_optimal_permutation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Get optimal permutation.

    Get the best permutation of positions to align two sets of points. It is assumed that
    the points are all of the same kind (ie, the same atomic type).

    The alignment is done by solving the optimal transport problem, using the squared geodesic
    distance as the cost.

    Args:
        x: a point on the hyper-torus, dimension [number_of_atoms, spatial_dimension].
        y: a point on the hyper-torus, dimension [number_of_atoms, spatial_dimension].

    Returns:
        optimal_permutation: the optimal permutation pi, such that pi.y is the shortest distance away from x.
    """
    cost_matrix = get_squared_geodesic_distance_cost_matrix(x, y)
    row_idx, col_idx = linear_sum_assignment(cost_matrix.to(torch.device('cpu')))
    n = cost_matrix.shape[0]
    optimal_permutation = torch.eye(n)[col_idx, :].to(cost_matrix.device)
    return optimal_permutation

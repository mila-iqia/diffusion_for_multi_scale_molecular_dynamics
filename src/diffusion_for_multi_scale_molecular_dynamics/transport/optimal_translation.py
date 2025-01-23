r"""Optimal Translation.

The code in this module aims to find the optimal global translation that minimizes the geodesic squared distance
between two points on the hyper-torus. The geodesic squared distance is piecewise quadratic and based on ATAN2.

Specifically, for n particles in d spatial dimension, let

    X = [---  x1 ---]     and      Y = [---  y1 ---]
        [---  x2 ---]                  [---  y2 ---]
        [--- ... ---]                  [--- ... ---]
        [---  xn ---]                  [---  yn ---]

where xi, yj are d dimensional arrays.

We seek a global translation of the form

        Tau = [--- tau ---]  (tau is a d dimensional array)
              [--- tau ---]
              [--- ... ---]
              [--- tau ---]

such that the squared geodesic distance D^2(X, Y + Tau) is minimized.

The algorithm below uses the fact that the squared geodesic distance D2 is piecewise quadratic;
we can thus find all its minima by setting its derivative with respect to tau^alpha to zero. This leads to a
self-consistent equation of the form

    tau^alpha = - 1 / n sum_{i=1}^{n}(yi^alpha - xi^alpha) + 1 / n sum_{i=1}^{n} l_i^\alpha(tau^alpha)

    l_i^alpha(tau^alpha) = Round(y_i^alpha - x_i^alpha + tau^alpha)

for alpha = 1, ..., d.

We can simply tabulate all the values of tau^alpha where the left-hand-side of the self-consistency equation
equals the right-hand-side, and then brute force identify which value has the smallest distance.

We will assume that -0.5 <= \tau^alpha <= 0.5.

The principal method in this module is :

    find_squared_geodesic_distance_minimizing_translation
"""

from typing import Tuple

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.transport.distance import \
    get_geodesic_displacements

TAU_RANGE_MIN = -0.5
TAU_RANGE_MAX = 0.5


def compute_integer_ells_and_tau_crossing_points(
    y_minus_x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute integer ells and tau crossing points.

    Using the relationship

        l_i^alpha(tau^alpha) = Round(y_i^alpha - x_i^alpha + tau^alpha),

    this method computes l_i^alpha(tau^alpha = TAU_RANGE_MIN), and the values of tau^alpha at which the value of ell is
    incremented by one.

    Args:
        y_minus_x: a tensor containing values of y_i^alpha - x_i^alpha

    Returns:
        l0: the values of l_i^alpha(tau^alpha = TAU_RANGE_MIN)
        tau_crossings: the values of tau^alpha at which l_i^alpha -> l_i^alpha + 1.
    """
    l0 = torch.round(y_minus_x + TAU_RANGE_MIN)
    epsilons = y_minus_x - l0 + TAU_RANGE_MIN
    tau_crossings = -epsilons

    return l0, tau_crossings


def get_plateau_values_and_boundaries(
    l0: torch.Tensor, tau_crossings: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get plateau values and starting tau values.

    Args:
        l0: the values of l_i^alpha(tau^alpha= TAU_RANGE_MIN), of
            dimensions [batch_size, number_of_atoms, spatial_dimension].
        tau_crossings: the values of tau^alpha at which l_i^alpha -> l_i^alpha + 1,
            of dimensions [batch_size, number_of_atoms, spatial_dimension].

    Returns:
        l_plateaus: the values of sum_{i=1}^{n} l_i^alpha(tau^alpha), which forms discrete plateaus.
            An array of dimension [number_of_atoms + 1, spatial_dimension].
        plateau_left_tau_values: the values of tau^alpha at which a plateau begins, always starting with TAU_RANGE_MIN.
        plateau_right_tau_values: the values of tau^alpha at which a plateau end, always ending with TAU_RANGE_MAX.
    """
    assert (
        len(l0.shape) == 3 and len(tau_crossings.shape) == 3
    ), "The input tensors should have 3 dimensions, [batch_size, number_of_atoms, spatial_dimension]."
    batch_size, number_of_atoms, spatial_dimension = tau_crossings.shape
    # Sort along the atom dimension. This will be the set of tau values at which l(tau) has a jump.
    sorted_tau_crossings = tau_crossings.sort(dim=1).values

    # Create arrays that capture the tau boundaries of the plateaus.
    # There are  "number_of_atoms + 1" plateaus
    starting_tau = TAU_RANGE_MIN * torch.ones(batch_size, 1, spatial_dimension)
    ending_tau = TAU_RANGE_MAX * torch.ones(batch_size, 1, spatial_dimension)

    plateau_left_tau_values = torch.cat([starting_tau, sorted_tau_crossings], dim=1)
    plateau_right_tau_values = torch.cat(
        [plateau_left_tau_values[:, 1:, :], ending_tau], dim=1
    )

    # The values of l for tau = TAU_RANGE_MIN
    starting_total_l = l0.sum(dim=1, keepdim=True)

    l_plateaus = (sorted_tau_crossings > TAU_RANGE_MIN).cumsum(dim=1) + starting_total_l
    l_plateaus = torch.cat([starting_total_l, l_plateaus], dim=1)

    return l_plateaus, plateau_left_tau_values, plateau_right_tau_values


def find_self_consistent_taus(
    y_minus_x: torch.Tensor,
) -> Tuple[torch.tensor, torch.Tensor, torch.Tensor]:
    """Find self-consistent taus.

    Args:
        y_minus_x: a tensor containing values of y_i^alpha - x_i^alpha,
            of dimensions [batch_size, number_of_atoms, spatial_dimension]

    Returns:
        tau_alphas:  all solutions to the self-consistent equation, as a one dimensional tensor.
        batch_indices: the value of batch index for each entry in the tensor tau_alphas.
        alphas: the value of alpha for each entry in the tensor tau_alphas.
    """
    assert (
        len(y_minus_x.shape) == 3
    ), "The input tensor should have 3 dimensions, [batch_size, number_of_atoms, spatial_dimension]."
    batch_size, number_of_atoms, spatial_dimension = y_minus_x.shape
    l0, tau_crossings = compute_integer_ells_and_tau_crossing_points(y_minus_x)

    l_plateaus, plateau_left_tau_values, plateau_right_tau_values = (
        get_plateau_values_and_boundaries(l0, tau_crossings)
    )

    euclidian_center_of_mass = einops.repeat(
        y_minus_x.mean(dim=1), "b d -> b n d", n=number_of_atoms + 1
    )

    right_hand_side = l_plateaus / number_of_atoms - euclidian_center_of_mass

    # find points where the right hand size is equal to the identity line
    self_consistent_solution_mask = torch.logical_and(
        right_hand_side > plateau_left_tau_values,
        right_hand_side < plateau_right_tau_values,
    )

    all_batch_indices = einops.repeat(
        torch.arange(batch_size),
        "b -> b n d",
        d=spatial_dimension,
        n=number_of_atoms + 1,
    )
    all_alpha = einops.repeat(
        torch.arange(spatial_dimension),
        "d -> b n d",
        b=batch_size,
        n=number_of_atoms + 1,
    )

    batch_indices = torch.masked_select(
        all_batch_indices, self_consistent_solution_mask
    )
    alphas = torch.masked_select(all_alpha, self_consistent_solution_mask)
    tau_alphas = torch.masked_select(right_hand_side, self_consistent_solution_mask)

    return tau_alphas, batch_indices, alphas


def find_squared_geodesic_distance_minimizing_translation(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Find squared geodesic distance minimizing translation.

    Args:
        x: a batch of points on the hyper-torus, dimension [batch_size, number_of_atoms, spatial_dimension].
        y: a batch of points on the hyper-torus, dimension [batch_size, number_of_atoms, spatial_dimension].

    Returns:
        minimum_tau: global translation that minimizes the squared geodesic distance D2(x, y + tau).
            Dimension [batch_size, spatial_dimension].
    """
    assert len(x.shape) == 3 and len(
        y.shape
    ), "The input tensor should have 3 dimensions, [batch_size, number_of_atoms, spatial_dimension]."

    batch_size, number_of_atoms, spatial_dimension = x.shape

    # Arrays of dimension [total_number_of_candidates], which is all the candidates for
    # all spatial dimensions and all batch elements.
    tau_alphas, batch_indices, alphas = find_self_consistent_taus(y - x)

    number_of_candidates = len(alphas)

    # Broadcast the columns according to the required dimensions
    # to dimension [total_number_of_candidates, number_of_atoms]
    x_alphas = x[batch_indices, :, alphas]
    y_alphas = y[batch_indices, :, alphas]

    y_plus_tau_alphas = y_alphas + einops.repeat(
        tau_alphas, "t ->  t n", n=number_of_atoms
    )

    # compute all one-dimensional squared distances
    componentwise_squared_distance = (
        get_geodesic_displacements(x_alphas, y_plus_tau_alphas) ** 2
    )

    # sum on atoms to get candidate minimum squared distances
    minimum_value_candidates = componentwise_squared_distance.sum(dim=1)

    # cast as high dimensional matrices to simplify the extraction of optimal values of tau.
    tau_matrix = torch.inf * torch.ones(
        number_of_candidates, batch_size, spatial_dimension
    )
    tau_matrix[torch.arange(number_of_candidates), batch_indices, alphas] = tau_alphas

    candidate_squared_distance_matrix = torch.inf * torch.ones(
        number_of_candidates, batch_size, spatial_dimension
    )
    candidate_squared_distance_matrix[
        torch.arange(number_of_candidates), batch_indices, alphas
    ] = minimum_value_candidates

    # For each batch element and spatial dimension, identify the index of the smallest squared distance.
    candidate_indices = candidate_squared_distance_matrix.argmin(dim=0)

    # Extract the corresponding value of tau
    minimum_tau = torch.gather(
        input=tau_matrix, dim=0, index=candidate_indices.unsqueeze(0)
    ).squeeze(0)

    return minimum_tau

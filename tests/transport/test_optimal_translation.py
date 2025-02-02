import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.transport.distance import \
    get_squared_geodesic_distance
from diffusion_for_multi_scale_molecular_dynamics.transport.optimal_translation import (
    TAU_RANGE_MAX, TAU_RANGE_MIN, compute_integer_ells_and_tau_crossing_points,
    find_squared_geodesic_distance_minimizing_translation,
    get_plateau_values_and_boundaries)


@pytest.fixture()
def batch_size():
    return 2


@pytest.fixture()
def number_of_atoms():
    return 4


@pytest.fixture()
def spatial_dimension():
    return 3


@pytest.fixture()
def x(batch_size, number_of_atoms, spatial_dimension, device):
    return torch.rand(batch_size, number_of_atoms, spatial_dimension).to(device)


@pytest.fixture()
def y(batch_size, number_of_atoms, spatial_dimension, device):
    return torch.rand(batch_size, number_of_atoms, spatial_dimension).to(device)


def test_compute_integer_ells_and_tau_crossing_points(x, y):
    y_minus_x = y - x

    l0, tau_crossings = compute_integer_ells_and_tau_crossing_points(y_minus_x)

    torch.testing.assert_close(torch.round(y_minus_x + TAU_RANGE_MIN), l0)

    small = 1.0e-6
    l_above_crossings = torch.round(y_minus_x + tau_crossings + small)
    l_below_crossings = torch.round(y_minus_x + tau_crossings - small)

    torch.testing.assert_close(l_below_crossings, l0)
    torch.testing.assert_close(l_above_crossings, l0 + 1)


def test_get_plateau_values_and_boundaries(x, y, batch_size, spatial_dimension, device):

    l0, tau_crossings = compute_integer_ells_and_tau_crossing_points(y - x)
    l_plateaus, plateau_left_tau_values, plateau_right_tau_values = (
        get_plateau_values_and_boundaries(l0, tau_crossings)
    )

    for batch_idx in torch.arange(batch_size):
        for alpha in torch.arange(spatial_dimension):
            list_l0 = l0[batch_idx, :, alpha]
            list_tau_crossings = tau_crossings[batch_idx, :, alpha]

            expected_left_tau_values = [TAU_RANGE_MIN]
            expected_l_plateaus = [torch.sum(list_l0)]
            for tau in list_tau_crossings.sort().values:
                expected_left_tau_values.append(tau)
                expected_l_plateaus.append(expected_l_plateaus[-1] + 1.0)

            expected_l_plateaus = torch.tensor(expected_l_plateaus).to(device)
            expected_left_tau_values = torch.tensor(expected_left_tau_values).to(device)
            expected_right_tau_values = torch.hstack(
                [expected_left_tau_values[1:], torch.tensor(TAU_RANGE_MAX).to(device)]
            )

            torch.testing.assert_close(
                expected_l_plateaus, l_plateaus[batch_idx, :, alpha]
            )
            torch.testing.assert_close(
                expected_left_tau_values, plateau_left_tau_values[batch_idx, :, alpha]
            )
            torch.testing.assert_close(
                expected_right_tau_values, plateau_right_tau_values[batch_idx, :, alpha]
            )


def test_find_squared_geodesic_distance_minimizing_translation(
    x, y, batch_size, spatial_dimension
):

    tau_minimum = find_squared_geodesic_distance_minimizing_translation(x, y)

    list_t = torch.linspace(TAU_RANGE_MIN, TAU_RANGE_MAX, 1001)
    dt = list_t[1] - list_t[0]

    # Find approximate minima by brute force: make sure the answer is within a grid spacing
    # of the brute force value.
    for batch_idx in torch.arange(batch_size):
        for alpha in torch.arange(spatial_dimension):
            d2 = torch.tensor(
                [
                    get_squared_geodesic_distance(
                        x[batch_idx, :, alpha], y[batch_idx, :, alpha] + t
                    )
                    for t in list_t
                ]
            )
            approximate_minimum_tau = list_t[d2.argmin()]
            computed_tau_minimum = tau_minimum[batch_idx, alpha]

            assert (
                approximate_minimum_tau - dt
                <= computed_tau_minimum
                <= approximate_minimum_tau + dt
            )

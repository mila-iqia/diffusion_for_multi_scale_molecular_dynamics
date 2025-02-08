import einops
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from diffusion_for_multi_scale_molecular_dynamics.transport.distance import \
    get_squared_geodesic_distance_cost_matrix
from diffusion_for_multi_scale_molecular_dynamics.transport.transporter import \
    Transporter
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from diffusion_for_multi_scale_molecular_dynamics.utils.geometric_utils import \
    get_cubic_point_group_symmetries


@pytest.fixture()
def batch_size():
    return 8


@pytest.fixture()
def number_of_atoms():
    return 6


@pytest.fixture()
def spatial_dimension():
    return 3


@pytest.fixture()
def x(batch_size, number_of_atoms, spatial_dimension, device):
    return torch.rand(batch_size, number_of_atoms, spatial_dimension).to(device)


@pytest.fixture()
def mu(batch_size, number_of_atoms, spatial_dimension, device):
    return torch.rand(batch_size, number_of_atoms, spatial_dimension).to(device)


@pytest.fixture(params=[True, False])
def use_point_groups(request):
    return request.param


@pytest.fixture()
def point_group_operations(spatial_dimension, use_point_groups, device):
    if use_point_groups:
        point_group_symmetries = get_cubic_point_group_symmetries(spatial_dimension).to(device)
    else:
        point_group_symmetries = torch.eye(spatial_dimension).unsqueeze(0).to(device)

    return point_group_symmetries


@pytest.fixture()
def random_point_group_operations(batch_size, point_group_operations, device):
    indices = torch.randint(0, len(point_group_operations), (batch_size,))
    return point_group_operations[indices].to(device)


@pytest.fixture()
def random_global_translations(batch_size, number_of_atoms, spatial_dimension, device):
    tau = einops.repeat(
        torch.rand(batch_size, spatial_dimension), "b d -> b n d", n=number_of_atoms
    ).to(device)
    return tau


@pytest.fixture()
def random_permutations(batch_size, number_of_atoms, device):
    permutations = []
    for _ in range(batch_size):
        idx = torch.randperm(number_of_atoms)
        permutations.append(torch.eye(number_of_atoms)[idx, :])
    permutations = torch.stack(permutations)
    return permutations.to(device)


@pytest.fixture()
def transporter(point_group_operations):
    return Transporter(point_group_operations)


def test_find_pseudo_center_of_mass(transporter, x, batch_size, spatial_dimension):
    computed_tau = transporter._find_pseudo_center_of_mass(x)
    assert computed_tau.shape == (batch_size, spatial_dimension)


def test_substact_center_of_mass(transporter, x):
    x_com = transporter._find_pseudo_center_of_mass(x)
    x_minus_x_com = transporter._substact_center_of_mass(x, x_com)

    should_be_zero = transporter._find_pseudo_center_of_mass(x_minus_x_com)

    # Because of numerical bjorks, the results can also be very close to 1, which is equivalent to zero.
    torch.testing.assert_close(
        should_be_zero - torch.round(should_be_zero), torch.zeros_like(should_be_zero)
    )


def test_get_all_cost_matrices(transporter, x, mu, batch_size, point_group_operations):
    x_com = transporter._find_pseudo_center_of_mass(x)
    mu_com = transporter._find_pseudo_center_of_mass(mu)
    x_minus_x_com = transporter._substact_center_of_mass(x, x_com)
    mu_minus_mu_com = transporter._substact_center_of_mass(mu, mu_com)

    computed_cost_matrices = transporter._get_all_cost_matrices(
        x_minus_x_com, mu_minus_mu_com
    )

    for batch_idx in range(batch_size):
        x1 = x_minus_x_com[batch_idx]
        x2 = mu_minus_mu_com[batch_idx]
        for op_idx, op in enumerate(point_group_operations):
            rot_x2 = einops.einsum(op, x2, "d1 d2, n d2 -> n d1")
            expected_cost_matrix = get_squared_geodesic_distance_cost_matrix(x1, rot_x2)
            torch.testing.assert_close(
                computed_cost_matrices[batch_idx, op_idx], expected_cost_matrix
            )


def test_find_permutation_and_cost(transporter, number_of_atoms):

    random_cost_matrix = torch.rand(number_of_atoms, number_of_atoms)

    permutation, computed_cost = transporter._find_permutation_and_cost(
        random_cost_matrix
    )
    expected_cost = torch.matmul(permutation, random_cost_matrix).trace()

    torch.testing.assert_close(computed_cost, expected_cost)


def test_solve_linear_assigment_problem(
    transporter, x, mu, batch_size, point_group_operations
):
    small_epsilon = 1.0e-6

    x_com = transporter._find_pseudo_center_of_mass(x)
    mu_com = transporter._find_pseudo_center_of_mass(mu)
    x_minus_x_com = transporter._substact_center_of_mass(x, x_com)
    mu_minus_mu_com = transporter._substact_center_of_mass(mu, mu_com)

    computed_cost_matrices = transporter._get_all_cost_matrices(
        x_minus_x_com, mu_minus_mu_com
    )

    best_permutations, best_point_group_operations = (
        transporter._solve_linear_assigment_problem(computed_cost_matrices)
    )

    for batch_idx in range(batch_size):
        permutation = best_permutations[batch_idx]
        op = best_point_group_operations[batch_idx]

        op_idx = torch.sum((point_group_operations - op) ** 2, dim=[1, 2]).argmin()

        cost_matrix = computed_cost_matrices[batch_idx, op_idx]
        computed_lowest_cost = torch.matmul(permutation, cost_matrix).trace()

        # Test that the computed permutation leads to the correct cost.
        row_idx, col_idx = linear_sum_assignment(cost_matrix.to(torch.device("cpu")))
        expected_cost = cost_matrix[row_idx, col_idx].sum()
        torch.testing.assert_close(computed_lowest_cost, expected_cost)

        # Test that the computed lowest cost is indeed the minimum. Allow some slack in the test
        # to account for possible numerical error.
        list_possible_costs = []
        for possible_cost_matrix in computed_cost_matrices[batch_idx]:
            row_idx, col_idx = linear_sum_assignment(
                possible_cost_matrix.to(torch.device("cpu"))
            )
            possible_cost = cost_matrix[row_idx, col_idx].sum()
            list_possible_costs.append(possible_cost)
        list_possible_costs = torch.stack(list_possible_costs)
        assert torch.all(computed_lowest_cost <= list_possible_costs + small_epsilon)


def test_get_optimal_transport(
    transporter,
    x,
    mu,
    random_point_group_operations,
    random_permutations,
    random_global_translations,
):

    # Test that the transport is equivariant
    best_mu_image = transporter.get_optimal_transport(x, mu)

    rot_mu = einops.einsum(
        random_point_group_operations, best_mu_image, "b d1 d2, b n d2 -> b n d1"
    )
    perm_rot_mu = einops.einsum(
        random_permutations, rot_mu, "b n1 n2, b n2 d -> b n1 d"
    )
    transported_best_mu_image = map_relative_coordinates_to_unit_cell(
        perm_rot_mu + random_global_translations
    )

    rot_x = einops.einsum(random_point_group_operations, x, "b d1 d2, b n d2 -> b n d1")
    perm_rot_x = einops.einsum(random_permutations, rot_x, "b n1 n2, b n2 d -> b n1 d")
    transported_x = map_relative_coordinates_to_unit_cell(
        perm_rot_x + random_global_translations
    )

    expected_transported_best_mu_image = transporter.get_optimal_transport(
        transported_x, mu
    )

    torch.testing.assert_allclose(
        transported_best_mu_image, expected_transported_best_mu_image
    )

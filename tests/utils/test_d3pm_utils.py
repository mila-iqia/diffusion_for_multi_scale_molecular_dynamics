import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import (
    class_index_to_onehot, compute_q_at_given_a0, compute_q_at_given_atm1,
    get_probability_at_previous_time_step)
from diffusion_for_multi_scale_molecular_dynamics.utils.tensor_utils import \
    broadcast_batch_matrix_tensor_to_all_dimensions


@pytest.fixture(scope="module", autouse=True)
def set_random_seed():
    torch.manual_seed(2345234)


@pytest.fixture()
def final_shape(batch_size, number_of_dimensions):
    shape = torch.randint(low=1, high=5, size=(number_of_dimensions,))
    shape[0] = batch_size
    return tuple(shape.numpy())


@pytest.fixture()
def batch_values(final_shape, num_classes):
    return torch.randint(0, num_classes, final_shape)


@pytest.fixture()
def q_t(final_shape, num_classes):
    return torch.randn(final_shape + (num_classes, num_classes))


@pytest.fixture()
def one_hot_x(batch_values, num_classes):
    return torch.nn.functional.one_hot(batch_values.long(), num_classes)


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("number_of_dimensions", [4, 8])
@pytest.mark.parametrize("num_classes", [1, 2, 3])
def test_class_index_to_onehot(batch_size, batch_values, final_shape, num_classes):
    computed_onehot_encoded = class_index_to_onehot(batch_values, num_classes)

    expected_encoding = torch.zeros(final_shape + (num_classes,))
    for i in range(num_classes):
        expected_encoding[..., i] += torch.where(batch_values == i, 1, 0)
    assert torch.all(expected_encoding == computed_onehot_encoded)


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("number_of_dimensions", [4, 8])
@pytest.mark.parametrize("num_classes", [1, 2, 3])
def test_compute_q_xt_bar_xo(q_t, one_hot_x, num_classes):
    computed_q_xtxo = compute_q_at_given_a0(one_hot_x, q_t)
    expected_q_xtxo = torch.zeros_like(one_hot_x.float())
    for i in range(num_classes):
        for j in range(num_classes):
            expected_q_xtxo[..., i] += one_hot_x[..., j].float() * q_t[..., j, i]
    torch.testing.assert_allclose(computed_q_xtxo, expected_q_xtxo)


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("number_of_dimensions", [4, 8])
@pytest.mark.parametrize("num_classes", [1, 2, 3])
def test_compute_q_xt_bar_xtm1(q_t, one_hot_x, num_classes):
    computed_q_xtxtm1 = compute_q_at_given_atm1(one_hot_x, q_t)
    expected_q_xtxtm1 = torch.zeros_like(one_hot_x.float())
    for i in range(num_classes):
        for j in range(num_classes):
            expected_q_xtxtm1[..., i] += one_hot_x[..., j].float() * q_t[..., j, i]
    torch.testing.assert_allclose(computed_q_xtxtm1, expected_q_xtxtm1)


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def number_of_atoms():
    return 8


@pytest.fixture
def num_atom_types():
    return 5


@pytest.fixture
def num_classes(num_atom_types):
    return num_atom_types + 1


@pytest.fixture
def predicted_logits(batch_size, number_of_atoms, num_classes):
    logits = 10 * (torch.randn(batch_size, number_of_atoms, num_classes) - 0.5)
    logits[:, :, -1] = -torch.inf  # force the model to never predict MASK
    return logits


@pytest.fixture
def predicted_p_a0_given_at(predicted_logits):
    return torch.nn.functional.softmax(predicted_logits, dim=-1)


@pytest.fixture
def one_hot_at(batch_size, number_of_atoms, num_atom_types, num_classes):
    # at CAN be MASK.
    one_hot_indices = torch.randint(
        0,
        num_classes,
        (
            batch_size,
            number_of_atoms,
        ),
    )
    one_hots = class_index_to_onehot(one_hot_indices, num_classes=num_classes)
    return one_hots


@pytest.fixture
def q_matrices(batch_size, number_of_atoms, num_classes):
    random_q_matrices = torch.rand(batch_size, num_classes, num_classes)
    final_shape = (batch_size, number_of_atoms)
    broadcast_q_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
        random_q_matrices, final_shape=final_shape
    )
    return broadcast_q_matrices


@pytest.fixture
def q_bar_matrices(batch_size, number_of_atoms, num_classes):
    random_q_bar_matrices = torch.rand(batch_size, num_classes, num_classes)
    final_shape = (batch_size, number_of_atoms)
    broadcast_q_bar_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
        random_q_bar_matrices, final_shape=final_shape
    )
    return broadcast_q_bar_matrices


@pytest.fixture
def q_bar_tm1_matrices(batch_size, number_of_atoms, num_classes):
    random_q_bar_tm1_matrices = torch.rand(batch_size, num_classes, num_classes)
    final_shape = (batch_size, number_of_atoms)
    broadcast_q_bar_tm1_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
        random_q_bar_tm1_matrices, final_shape=final_shape
    )
    return broadcast_q_bar_tm1_matrices


@pytest.fixture
def loss_eps():
    return 1.0e-12


@pytest.fixture
def expected_p_atm1_given_at_from_logits(
    predicted_p_a0_given_at,
    one_hot_at,
    q_matrices,
    q_bar_matrices,
    q_bar_tm1_matrices,
):
    batch_size, natoms, num_classes = predicted_p_a0_given_at.shape

    denominator = torch.zeros(batch_size, natoms)
    numerator1 = torch.zeros(batch_size, natoms, num_classes)
    numerator2 = torch.zeros(batch_size, natoms, num_classes)

    for i in range(num_classes):
        for j in range(num_classes):
            denominator[:, :] += (
                predicted_p_a0_given_at[:, :, i]
                * q_bar_matrices[:, :, i, j]
                * one_hot_at[:, :, j]
            )
            numerator1[:, :, i] += (
                predicted_p_a0_given_at[:, :, j] * q_bar_tm1_matrices[:, :, j, i]
            )
            numerator2[:, :, i] += q_matrices[:, :, i, j] * one_hot_at[:, :, j]

    numerator = numerator1 * numerator2

    expected_p = torch.zeros(batch_size, natoms, num_classes)
    for i in range(num_classes):
        expected_p[:, :, i] = numerator[:, :, i] / denominator[:, :]

    # Note that the expected_p_atm1_given_at is not really a probability (and thus does not sum to 1) because
    # the Q matrices are random.
    return expected_p


@pytest.fixture
def one_hot_a0(batch_size, number_of_atoms, num_atom_types, num_classes):
    # a0 CANNOT be MASK.
    one_hot_indices = torch.randint(
        0,
        num_atom_types,
        (
            batch_size,
            number_of_atoms,
        ),
    )
    one_hots = class_index_to_onehot(one_hot_indices, num_classes=num_classes)
    return one_hots


@pytest.fixture
def expected_p_atm1_given_at_from_onehot(
    one_hot_a0, one_hot_at, q_matrices, q_bar_matrices, q_bar_tm1_matrices
):
    batch_size, natoms, num_classes = one_hot_a0.shape

    denominator = torch.zeros(batch_size, natoms)
    numerator1 = torch.zeros(batch_size, natoms, num_classes)
    numerator2 = torch.zeros(batch_size, natoms, num_classes)

    for i in range(num_classes):
        for j in range(num_classes):
            denominator[:, :] += (
                one_hot_a0[:, :, i] * q_bar_matrices[:, :, i, j] * one_hot_at[:, :, j]
            )
            numerator1[:, :, i] += one_hot_a0[:, :, j] * q_bar_tm1_matrices[:, :, j, i]
            numerator2[:, :, i] += q_matrices[:, :, i, j] * one_hot_at[:, :, j]

    numerator = numerator1 * numerator2

    expected_q = torch.zeros(batch_size, natoms, num_classes)
    for i in range(num_classes):
        expected_q[:, :, i] = numerator[:, :, i] / denominator[:, :]

    return expected_q


def test_get_probability_at_previous_time_step_from_logits(
    predicted_logits,
    one_hot_at,
    q_matrices,
    q_bar_matrices,
    q_bar_tm1_matrices,
    loss_eps,
    expected_p_atm1_given_at_from_logits,
):
    computed_p_atm1_given_at = get_probability_at_previous_time_step(
        predicted_logits,
        one_hot_at,
        q_matrices,
        q_bar_matrices,
        q_bar_tm1_matrices,
        small_epsilon=loss_eps,
        probability_at_zeroth_timestep_are_logits=True,
    )

    assert torch.allclose(
        computed_p_atm1_given_at, expected_p_atm1_given_at_from_logits
    )


def test_get_probability_at_previous_time_step_from_one_hot_probabilities(
    one_hot_a0,
    one_hot_at,
    q_matrices,
    q_bar_matrices,
    q_bar_tm1_matrices,
    loss_eps,
    expected_p_atm1_given_at_from_onehot,
):
    computed_q_atm1_given_at_and_a0 = get_probability_at_previous_time_step(
        one_hot_a0,
        one_hot_at,
        q_matrices,
        q_bar_matrices,
        q_bar_tm1_matrices,
        small_epsilon=loss_eps,
        probability_at_zeroth_timestep_are_logits=False,
    )

    assert torch.allclose(
        computed_q_atm1_given_at_and_a0, expected_p_atm1_given_at_from_onehot
    )

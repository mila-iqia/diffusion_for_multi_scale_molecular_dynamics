import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import (
    class_index_to_onehot, compute_q_at_given_a0, compute_q_at_given_atm1)


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

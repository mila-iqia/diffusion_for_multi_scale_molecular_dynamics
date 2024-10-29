import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.tensor_utils import (
    broadcast_batch_matrix_tensor_to_all_dimensions,
    broadcast_batch_tensor_to_all_dimensions,
)


@pytest.fixture(scope="module", autouse=True)
def set_random_seed():
    torch.manual_seed(2345234)


@pytest.fixture()
def batch_values(batch_size):
    return torch.rand(batch_size)


@pytest.fixture()
def batch_matrix_values(batch_size, num_classes):
    return torch.rand(batch_size, num_classes, num_classes)


@pytest.fixture()
def final_shape(batch_size, number_of_dimensions):
    shape = torch.randint(low=1, high=5, size=(number_of_dimensions,))
    shape[0] = batch_size
    return tuple(shape.numpy())


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("number_of_dimensions", [1, 2, 3])
def test_broadcast_batch_tensor_to_all_dimensions(
    batch_size, batch_values, final_shape
):
    broadcast_values = broadcast_batch_tensor_to_all_dimensions(
        batch_values, final_shape
    )

    value_arrays = broadcast_values.reshape(batch_size, -1)

    for expected_value, computed_values in zip(batch_values, value_arrays):
        expected_values = torch.ones_like(computed_values) * expected_value
        torch.testing.assert_close(expected_values, computed_values)


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("number_of_dimensions", [1, 2, 3])
@pytest.mark.parametrize("num_classes", [1, 2, 4])
def test_broadcast_batch_matrix_tensor_to_all_dimensions(
    batch_size, batch_matrix_values, final_shape, num_classes
):
    broadcast_values = broadcast_batch_matrix_tensor_to_all_dimensions(
        batch_matrix_values, final_shape
    )

    value_arrays = broadcast_values.reshape(batch_size, -1, num_classes, num_classes)

    for expected_value, computed_values in zip(batch_matrix_values, value_arrays):
        expected_values = torch.ones_like(computed_values) * expected_value
        torch.testing.assert_close(expected_values, computed_values)

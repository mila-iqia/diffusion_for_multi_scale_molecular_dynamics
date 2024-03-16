import pytest
import torch

from crystal_diffusion.utils.tensor_utils import \
    broadcast_batch_tensor_to_all_dimensions


@pytest.fixture()
def batch_values(batch_size):
    torch.manual_seed(2345234)
    return torch.rand(batch_size)


@pytest.fixture()
def final_shape(batch_size, number_of_dimensions):
    shape = torch.randint(low=1, high=5, size=(number_of_dimensions,))
    shape[0] = batch_size
    return tuple(shape.numpy())


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("number_of_dimensions", [1, 2, 3])
def test_broadcast_batch_tensor_to_all_dimensions(batch_size, batch_values, final_shape):
    broadcast_values = broadcast_batch_tensor_to_all_dimensions(batch_values, final_shape)

    value_arrays = broadcast_values.reshape(batch_size, -1)

    for value, value_array in zip(batch_values, value_arrays):
        assert torch.all(torch.isclose(value_array, value))

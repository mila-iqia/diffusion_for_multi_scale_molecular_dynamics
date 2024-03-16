from typing import Tuple

import torch


def broadcast_batch_tensor_to_all_dimensions(batch_values: torch.Tensor, final_shape: Tuple[int]) -> torch.Tensor:
    """Broadcast batch tensor to all dimensions.

    A data batch is typically a tensor of shape [batch_size, n1, n2, ...] where n1, n2, etc constitute
    one example of the data. This method broadcasts a tensor of shape [batch_size] to a tensor of shape
    [batch_size, n1, n2, ...] where all the values for the non-batch dimension are equal to the value
    for the given batch index.

    This is useful when we want to multiply every value in the data example by the same number.

    Args:
        batch_values : values to be braodcasted, of shape [batch_size]
        final_shape : shape of the final tensor, [batch_size, n1, n2, ...]

    Returns:
        broadcast_values : tensor of shape [batch_size, n1, n2, ...], where all entries are identical
            along non-batch dimensions.
    """
    assert len(batch_values.shape) == 1, "The batch values should be a one-dimensional tensor."
    batch_size = len(batch_values)

    assert final_shape[0] == batch_size, "The final shape should have the batch_size as its first dimension."

    # reshape the batch_values array to have the same dimension as final_shape, with all values identical
    # for a given batch index.
    number_of_dimensions = len(final_shape)
    reshape_dimension = [-1] + (number_of_dimensions - 1) * [1]
    broadcast_values = batch_values.reshape(reshape_dimension).expand(final_shape)
    return broadcast_values

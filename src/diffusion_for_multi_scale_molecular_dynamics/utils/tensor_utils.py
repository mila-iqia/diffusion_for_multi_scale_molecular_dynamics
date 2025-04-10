from typing import Tuple

import torch


def broadcast_batch_tensor_to_all_dimensions(
    batch_values: torch.Tensor, final_shape: Tuple[int, ...]
) -> torch.Tensor:
    """Broadcast batch tensor to all dimensions.

    A data batch is typically a tensor of shape [batch_size, n1, n2, ...] where n1, n2, etc constitute
    one example of the data. This method broadcasts a tensor of shape [batch_size] to a tensor of shape
    [batch_size, n1, n2, ...] where all the values for the non-batch dimension are equal to the value
    for the given batch index.

    This is useful when we want to multiply every value in the data example by the same number.

    Args:
        batch_values : values to be broadcasted, of shape [batch_size]
        final_shape : shape of the final tensor, [batch_size, n1, n2, ...]

    Returns:
        broadcast_values : tensor of shape [batch_size, n1, n2, ...], where all entries are identical
            along non-batch dimensions.
    """
    assert (
        len(batch_values.shape) == 1
    ), "The batch values should be a one-dimensional tensor."
    batch_size = len(batch_values)

    assert (
        final_shape[0] == batch_size
    ), "The final shape should have the batch_size as its first dimension."

    # reshape the batch_values array to have the same dimension as final_shape, with all values identical
    # for a given batch index.
    number_of_dimensions = len(final_shape)
    reshape_dimension = [-1] + (number_of_dimensions - 1) * [1]
    broadcast_values = batch_values.reshape(reshape_dimension).expand(final_shape)
    return broadcast_values


def broadcast_batch_matrix_tensor_to_all_dimensions(
    batch_values: torch.Tensor, final_shape: Tuple[int, ...]
) -> torch.Tensor:
    """Broadcast batch tensor to all dimensions.

    A data matrix batch is typically a tensor of shape [batch_size, n1, n2, ..., m1, m2] where n1, n2, etc constitute
    one example of the data and m1 and m2 are the matrix dimensions. This method broadcasts a tensor of shape
    [batch_size, m1, m2] to a tensor of shape
    [batch_size, n1, n2, ..., m1, m2] where all the values for the non-batch and matrix dimensions are equal to the
    value for the given batch index and matrix element.

    This is useful when we want to multiply every value in the data example by the same matrix.

    Args:
        batch_values : values to be broadcasted, of shape [batch_size, m1, m2]
        final_shape : shape of the final tensor, excluding the matrix dimensions [batch_size, n1, n2, ...,]

    Returns:
        broadcast_values : tensor of shape [batch_size, n1, n2, ..., m1, m2], where all entries are identical
            along non-batch and non-matrix dimensions.
    """
    assert (
        len(batch_values.shape) == 3
    ), "The batch values should be a three-dimensional tensor."
    batch_size = batch_values.shape[0]
    matrix_size = batch_values.shape[-2:]

    assert (
        final_shape[0] == batch_size
    ), "The final shape should have the batch_size as its first dimension."

    # reshape the batch_values array to have the same dimension as final_shape, with all values identical
    # for a given batch index.
    number_of_dimensions = len(final_shape)
    reshape_dimension = (
        torch.Size([batch_size] + (number_of_dimensions - 1) * [1]) + matrix_size
    )
    broadcast_values = batch_values.reshape(reshape_dimension).expand(
        torch.Size(final_shape) + matrix_size
    )
    return broadcast_values

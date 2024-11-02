"""Common operations used for Discrete Diffusion."""

import einops
import torch


def class_index_to_onehot(index: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert a tensor of class indices to a one-hot representation.

    Args:
        index: index tensor to encode
        num_classes: total number of classes

    Returns:
        float tensor of 0s and 1s. The size is x.size() + (num_classes)
    """
    # the last .to() acts on the tensor type to avoid longs
    return torch.nn.functional.one_hot(index.long(), num_classes=num_classes).to(index)


def compute_q_at_given_a0(
    one_hot_a0: torch.Tensor, q_bar_t: torch.Tensor
) -> torch.Tensor:
    """Compute q(a_t | a_0).

    This is done by the vector-matrix product: a_0 \bar{Q}_t assuming a_0 is a one-hot vector or a distribution over
    different classes.

    Args:
        one_hot_x0: initial state (a_0). The last dimension should be the number of classes.
        q_bar_t: cumulative Markov transition matrix (\bar{Q}_t). The last 2 dimensions should be the number of classes.

    Returns:
        matrix-vector product between one_hot_x0 and q_bar_t that defines q(a_t | a_0)
    """
    return einops.einsum(one_hot_a0.to(q_bar_t), q_bar_t, "... j, ... j i -> ... i")


def compute_q_at_given_atm1(
    one_hot_atm1: torch.Tensor, q_tm1: torch.Tensor
) -> torch.Tensor:
    """Compute q(a_t | a_{t-1}).

    This is done by the vector-matrix product: a_{t-1} Q_{t-1}^T assuming a_{t-1} is a one-hot vector or a distribution
        over different classes. The transition matrix Q is a 1-step transition matrix.

    Args:
        one_hot_atm1: state (a_{t-1}). The last dimension should be the number of classes.
        q_tm1: Markov transition matrix (Q_{t-1}). The last 2 dimensions should be the number of classes.

    Returns:
        matrix-vector product between one_hot_atm1 and q_{t-1}^T that defines q(a_t | a_{t-1})
    """
    return einops.einsum(
        one_hot_atm1.to(q_tm1),
        torch.transpose(q_tm1, -2, -1),
        "... j, ... i j -> ... i",
    )

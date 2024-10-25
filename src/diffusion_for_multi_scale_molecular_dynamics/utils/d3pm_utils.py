"""Common operations used for Discrete Diffusion."""
import einops
import torch


def class_index_to_onehot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert a tensor of class indices to a one-hot representation.

    Args:
        x: long tensor to encode
        num_classes: total number of classes

    Returns:
        long tensor of 0s and 1s. The size is x.size() + (num_classes)
    """
    return torch.nn.functional.one_hot(x.long(), num_classes=num_classes)


def compute_q_xt_bar_xo(one_hot_x0: torch.Tensor, q_bar_t: torch.Tensor) -> torch.Tensor:
    """Compute q(x_t | x_0).

    This is done by the vector-matrix product: x_0 \bar{Q}_t assuming x_0 is a one-hot vector or a distribution over
    different classes.

    Args:
        one_hot_x0: initial state (x_0). The last dimension should be the number of classes.
        q_bar_t: cumulative Markov transition matrix (\bar{Q}_t). The last 2 dimensions should be the number of classes.

    Returns:
        matrix-vector product between one_hot_x0 and q_bar_t that defines q(x_t | x_0)
    """
    return einops.einsum(one_hot_x0, q_bar_t, "... j, ... j i -> ... i")


def compute_q_xt_bar_xtm1(one_hot_xt: torch.Tensor, q_t: torch.Tensor) -> torch.Tensor:
    """Compute q(x_t | x_{t-1}).

    This is done by the vector-matrix product: x_t Q_t^T assuming x_t is a one-hot vector or a distribution over
    different classes.

    Args:
        one_hot_xt: state (x_t). The last dimension should be the number of classes.
        q_t: Markov transition matrix (Q_t). The last 2 dimensions should be the number of classes.

    Returns:
        matrix-vector product between one_hot_xt and q_t^T that defines q(x_t | x_{t-1})
    """
    return einops.einsum(one_hot_xt, torch.transpose(q_t, -2, -1), "... j, ... i j -> ... i")

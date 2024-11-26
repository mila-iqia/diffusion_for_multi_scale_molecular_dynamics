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
    return torch.nn.functional.one_hot(index.long(), num_classes=num_classes).to(
        device=index.device, dtype=torch.float
    )


def compute_q_at_given_a0(
    one_hot_a0: torch.Tensor, q_bar_t: torch.Tensor
) -> torch.Tensor:
    r"""Compute :math:`q(a_t | a_0)`.

    This is done by the vector-matrix product: :math:`a_0 \bar{Q}_t` assuming a_0 is a one-hot vector or a distribution
    over different classes.

    Args:
        one_hot_x0: initial state (:math:`a_0`). The last dimension should be the number of classes.
        q_bar_t: cumulative Markov transition matrix (:math:`\bar{Q}_t`). The last 2 dimensions should be the number of
            classes.

    Returns:
        matrix-vector product between one_hot_x0 and q_bar_t that defines :math:`q(a_t | a_0)`
    """
    return einops.einsum(one_hot_a0.to(q_bar_t), q_bar_t, "... j, ... j i -> ... i")


def compute_q_at_given_atm1(
    one_hot_atm1: torch.Tensor, q_tm1: torch.Tensor
) -> torch.Tensor:
    r"""Compute :math:`q(a_t | a_{t-1})`.

    This is done by the vector-matrix product: :math:`a_{t-1} Q_{t-1}^T` assuming :math:`a_{t-1}` is a one-hot vector or
        a distribution over different classes. The transition matrix Q is a 1-step transition matrix.

    Args:
        one_hot_atm1: state (:math:`a_{t-1}`). The last dimension should be the number of classes.
        q_tm1: Markov transition matrix (:math:`Q_{t-1}`). The last 2 dimensions should be the number of classes.

    Returns:
        matrix-vector product between one_hot_atm1 and :math:`Q_{t-1}^T` that defines :math:`q(a_t | a_{t-1})`
    """
    return einops.einsum(
        one_hot_atm1.to(q_tm1),
        torch.transpose(q_tm1, -2, -1),
        "... j, ... i j -> ... i",
    )


def get_probability_at_previous_time_step(
    probability_at_zeroth_timestep: torch.Tensor,
    one_hot_probability_at_current_timestep: torch.Tensor,
    q_matrices: torch.Tensor,
    q_bar_matrices: torch.Tensor,
    q_bar_tm1_matrices: torch.Tensor,
    small_epsilon: float,
    probability_at_zeroth_timestep_are_logits: bool = False,
) -> torch.Tensor:
    r"""Compute :math:`P(a_{t-1} | a_t, \gamma_0)`.

    For given probability distribution :math:`\gamma_0` and a one-hot distribution :math:`a_t`.

    .. math::
        P(a_{t-1} | a_t, \gamma_0) = (\gamma_0^T \cdot \bar{Q}_{t-1} \cdot a_{t-1}) (a_{t-1}^T \cdot Q_t \cdot a_t) /
                                        (\gamma_0^T \cdot \bar{Q}_{t} \cdot a_t)

    Args:
        probability_at_zeroth_timestep: :math:`\gamma_0` a probability representation of a class type (one-hot
            distribution or normalized distribution), as a tensor with dimension
            [batch_size, number_of_atoms, num_classes]
        one_hot_probability_at_current_timestep: :math:`a_t` a one-hot representation of a class type at current time
            step, as a tensor with dimension [batch_size, number_of_atoms, num_classes]
         q_matrices: :math:`{Q}_{t}` transition matrices at current time step of dimension
            [batch_size, number_of_atoms, num_classes, num_classes].
        q_bar_matrices: :math:`\bar{Q}_{t}` one-shot transition matrices at current time step of dimension
            [batch_size, number_of_atoms, num_classes, num_classes].
        q_bar_tm1_matrices: :math:`\bar{Q}_{t-1}` one-shot transition matrices at previous time step of dimension
            [batch_size, number_of_atoms, num_classes, num_classes].
        small_epsilon: minimum value for the denominator, to avoid division by zero.
        probability_at_zeroth_timestep_are_logits: if True, assume the probability_at_zeroth_timestep do not sum to 1
            and use a softmax on the last dimension to normalize. If False, assume the probabilities are normalized.
            Defaults to False.

    Returns:
        one-step transition normalized probabilities of dimension [batch_size, number_of_atoms, num_type_atoms]
    """
    if probability_at_zeroth_timestep_are_logits:
        probability_at_zeroth_timestep = torch.nn.functional.softmax(
            probability_at_zeroth_timestep, dim=-1
        )

    numerator1 = einops.einsum(
        probability_at_zeroth_timestep, q_bar_tm1_matrices, "... j, ... j i -> ... i"
    )
    numerator2 = einops.einsum(
        q_matrices, one_hot_probability_at_current_timestep, "... i j, ... j -> ... i"
    )
    numerator = numerator1 * numerator2

    den1 = einops.einsum(
        q_bar_matrices,
        one_hot_probability_at_current_timestep,
        "... i j, ... j -> ... i",
    )
    den2 = einops.einsum(
        probability_at_zeroth_timestep, den1, "... j, ... j -> ..."
    ).clip(min=small_epsilon)

    denominator = einops.repeat(
        den2, "... -> ... num_classes", num_classes=numerator.shape[-1]
    )

    return numerator / denominator

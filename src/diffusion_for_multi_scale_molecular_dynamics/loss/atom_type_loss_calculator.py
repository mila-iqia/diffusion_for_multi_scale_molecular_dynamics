import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import \
    LossParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import (
    compute_q_at_given_a0, compute_q_at_given_atm1)


class D3PMLossCalculator(torch.nn.Module):
    """Class to calculate the discrete diffusion loss."""

    def __init__(self, loss_parameters: LossParameters):
        """Initialize method."""
        super().__init__()
        # weight of the cross-entropy component
        self.ce_weight = loss_parameters.atom_types_ce_weight
        self.eps = loss_parameters.atom_types_eps

    def kl_loss_term(
        self,
        predicted_logits: torch.Tensor,
        one_hot_real_atom_types: torch.Tensor,
        one_hot_noisy_atom_types: torch.Tensor,
        q_matrices: torch.Tensor,
        q_bar_matrices: torch.Tensor,
        q_bar_tm1_matrices: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the KL component of the loss.

        This corresponds to this:

        .. math::

            D_{KL}[q(a_{t-1} | a_t, a_0) || p_\theta(a_{t-1} | a_{t})]

        We are ignoring the t=1 case here as we will use a NLL loss instead.

        Args:
            predicted_logits: output of the score network estimating an unnormalized
                :math:`p(a_0 | a_t)` of dimension [batch_size, number_of_atoms, num_type_atoms] where num_type_atoms
                includes the MASK token  TODO check if we should have num_type_atoms
            one_hot_real_atom_types: real atom types :math:`a_0` in one-hot format of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_type_atoms]
            one_hot_noisy_atom_types: noisy atom types :math:`a_t` in one-hot format of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_type_atoms]
            q_matrices: one-step transition matrices :math:`Q_t` of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_type_atoms]
            q_bar_matrices: one-shot transition matrices :math:`\bar{Q}_t` of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_type_atoms]
            q_bar_tm1_matrices: one-shot transition matrices at previous step :math:`\bar{Q}_{t-1}` of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_type_atoms]. An identity matrix is used for t=0.

        Returns:
            torch.Tensor: unreduced KL loss of dimension [batch_size, number_of_atoms, num_type_atoms]
        """
        # start by computing q(a_{t−1}|at, a0) = q(a_t | a_{t-1}, a_0) q(a_{t-1} | a_0) / q(a_t | a_0)
        # q(a_t | a_{t-1}, a0) = q(a_t | a_{t-1}) = a_t Q_t^T  - beware  the transpose here
        q_at_given_atm1 = compute_q_at_given_atm1(one_hot_noisy_atom_types, q_matrices)
        # dimension of q_at_bar_atm1 : batch_size, number_of_atoms, num_type_atoms
        # q(a_{t-1} | a_0) = a_0 \bar{Q}_{t-1}
        q_atm1_given_a0 = compute_q_at_given_a0(
            one_hot_real_atom_types, q_bar_tm1_matrices
        )
        # dimension of q_atm1_bar_a0: batch_size, number_of_atoms, num_type_atoms
        # q(a_t | a_0) = a_0 \bar{Q}_t a_t^T
        q_at_given_a0 = compute_q_at_given_a0(one_hot_real_atom_types, q_bar_matrices)
        at_probability = einops.einsum(
            q_at_given_a0, one_hot_noisy_atom_types.float(), "... i , ... i -> ..."
        )

        # dimension of at_probability: batch_size, number_of_atoms
        posterior_q = (
            q_at_given_atm1
            * q_atm1_given_a0
            / at_probability.unsqueeze(-1).clip(min=self.eps)
        )  # clip at eps
        # the unsqueeze in the denominator is to allow a broadcasting
        # posterior q has dimension: batch_size, number_of_atoms, num_type_atoms

        # we now need to compute p_\theta(a_{t-1} | a_t) using
        # p_\theta(a_{t-1} | a_t) \propto \sum_{\tilde{a}_0} q(a_{t-1}, a_t | \tilde{a}_0)p_\theta(\tilde{a}_0, a_t)
        # \propto \sum_{\tilde{a}_0} a_t Q_t^T \circ \tilde{a}_0 \bar{Q}_{t-1} \circ p_\theta(\tilde{a}_0 | a_t)
        # this is equivalent to doing a_t Q_t^T \circ \bar{Q}_{t-1} p_\theta(a_t)
        # with a matrix multiplication in the last step
        # we add a softmax to convert the predictions to normalized probabilities
        p_atm1_at = self.get_p_atm1_at(
            predicted_logits, q_at_given_atm1, q_bar_tm1_matrices
        )

        # get the KL divergence between posterior and predicted prob
        # do not reduce (average) yet as we will replace the samples with t=1 with a NLL loss
        # input of kl_div should be log-probabilities - we add eps to avoid log(0)
        kl_loss = torch.nn.functional.kl_div(
            torch.log(p_atm1_at + self.eps), posterior_q, reduction="none"
        )
        return kl_loss

    @staticmethod
    def get_p_atm1_at(
        predicted_logits: torch.Tensor,
        q_at_bar_atm1: torch.Tensor,
        q_bar_tm1_matrices: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute p(a_{t-1} | a_t).

        .. math::
            p_\theta(a_{t-1} | a_t) \propto \sum_{\tilde{a}_0} q(a_{t-1}, a_t | \tilde{a}_0)p_\theta(\tilde{a}_0, a_t)

        Args:
            predicted_logits: output of the score network estimating an unnormalized
                :math:`p(a_0 | a_t)` of dimension [batch_size, number_of_atoms, num_type_atoms] where num_type_atoms
                includes the MASK token
            q_at_bar_atm1: conditional posterior :math: `q(a_t | a_{t-1}, a0)` as a tensor with dimension
                [batch_size, number_of_atoms, num_type_atoms]
            q_bar_tm1_matrices: one-shot transition matrices at previous step :math:`\bar{Q}_{t-1}` of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_type_atoms]. An identity matrix is used for t=0.

        Returns:
            one-step transition normalized probabilities of dimension [batch_size, number_of_atoms, num_type_atoms]
        """
        p_atm1_at = q_at_bar_atm1 * einops.einsum(
            q_bar_tm1_matrices,
            torch.nn.functional.softmax(predicted_logits, dim=-1),
            "... j i, ... j -> ... i",
        )  # TODO revisit this
        return p_atm1_at

    def calculate_unreduced_loss(
        self,
        predicted_logits: torch.Tensor,
        one_hot_real_atom_types: torch.Tensor,
        one_hot_noisy_atom_types: torch.Tensor,
        time_indices: torch.Tensor,
        q_matrices: torch.Tensor,
        q_bar_matrices: torch.Tensor,
        q_bar_tm1_matrices: torch.Tensor,
    ) -> torch.Tensor:
        r"""Calculate unreduced loss.

        The loss is given by:

        .. math::

             L_a = E_{a_0 ~ p_\textrm{data}} [ \sum_{t=2}^T E_{a_t ~ p_{t|0}[
                    [D_{KL}[q(a_{t-1} | a_t, a_0) || p_theta(a_{t-1} | a_{t}] - \lambda_CE log p_\theta(a_0 | a_t)]
                    - E_{a_1 ~ p_{t=1| 0}} log p_\theta(a_0 | a_1)]

        Args:
            predicted_logits: output of the score network estimating an unnormalized
                :math:`p(a_0 | a_t)` of dimension [batch_size, number_of_atoms, num_type_atoms] where num_type_atoms
                includes the MASK token  # TODO revisit the output size and the name num_type_atoms vs num_classes
            one_hot_real_atom_types: real atom types :math:`a_0` as one-hot vectors of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_type_atoms]
            one_hot_noisy_atom_types: noisy atom types :math:`a_t` as one-hot vectors of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_type_atoms]
            time_indices: time indices sampled of dimension [batch_size]
            q_matrices: one-step transition matrices :math:`Q_t` of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_type_atoms]
            q_bar_matrices: one-shot transition matrices :math:`\bar{Q}_t` of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_type_atoms]
            q_bar_tm1_matrices: one-shot transition matrices at previous step :math:`\bar{Q}_{t-1}` of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_type_atoms]. An identity matrix is used for t=0

        Returns:
            unreduced_loss: a tensor of shape [batch_size, number_of_atoms, num_type_atoms]. It's mean is the loss.
        """
        # D_{KL}[q(a_{t-1} | a_t, a_0) || p_\theta(a_{t-1} | a_{t}]
        kl_term = self.kl_loss_term(
            predicted_logits,
            one_hot_real_atom_types,
            one_hot_noisy_atom_types,
            q_matrices,
            q_bar_matrices,
            q_bar_tm1_matrices,
        )

        # -log p_\theta(a_0 | a_t)
        nll_term = -torch.nn.functional.log_softmax(predicted_logits, dim=-1)

        # if t == 1 (0 for python indexing convention), use the NLL term, otherwise use the KL + \lambda_{CE} NLL
        d3pm_loss = torch.where(
            time_indices.view(-1, 1, 1) == 0,
            nll_term,
            kl_term + self.ce_weight * nll_term,
        )
        return d3pm_loss

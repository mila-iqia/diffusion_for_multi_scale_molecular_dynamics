import torch

from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import \
    LossParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    get_probability_at_previous_time_step


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
            predicted_logits: output of the score network estimating class logits
                :math:`p(a_0 | a_t)` of dimension [batch_size, number_of_atoms, num_classes] where num_classes
                includes the MASK token
            one_hot_real_atom_types: real atom types :math:`a_0` in one-hot format of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_classes]
            one_hot_noisy_atom_types: noisy atom types :math:`a_t` in one-hot format of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_classes]
            q_matrices: one-step transition matrices :math:`Q_t` of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_classes]
            q_bar_matrices: one-shot transition matrices :math:`\bar{Q}_t` of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_classes]
            q_bar_tm1_matrices: one-shot transition matrices at previous step :math:`\bar{Q}_{t-1}` of dimension
                [batch_size, number_of_atoms, num_type_atoms, num_classes]. An identity matrix is used for t=0.

        Returns:
            torch.Tensor: unreduced KL loss of dimension [batch_size, number_of_atoms, num_classes]
        """
        # The posterior probabilities
        q_atm1_given_at_and_a0 = self.get_q_atm1_given_at_and_a0(
            one_hot_a0=one_hot_real_atom_types,
            one_hot_at=one_hot_noisy_atom_types,
            q_matrices=q_matrices,
            q_bar_matrices=q_bar_matrices,
            q_bar_tm1_matrices=q_bar_tm1_matrices,
            small_epsilon=self.eps,
        )

        # The predicted probabilities
        p_atm1_given_at = self.get_p_atm1_given_at(
            predicted_logits=predicted_logits,
            one_hot_at=one_hot_noisy_atom_types,
            q_matrices=q_matrices,
            q_bar_matrices=q_bar_matrices,
            q_bar_tm1_matrices=q_bar_tm1_matrices,
            small_epsilon=self.eps,
        )

        # get the KL divergence between posterior and predicted probabilities
        # do not reduce (average) yet as we will replace the samples with t=1 with a NLL loss
        # input of kl_div should be log-probabilities.
        log_p = torch.log(p_atm1_given_at.clip(min=self.eps))
        kl_loss = torch.nn.functional.kl_div(
            log_p, q_atm1_given_at_and_a0, reduction="none"
        )
        return kl_loss

    @classmethod
    def get_q_atm1_given_at_and_a0(
        cls,
        one_hot_a0: torch.Tensor,
        one_hot_at: torch.Tensor,
        q_matrices: torch.Tensor,
        q_bar_matrices: torch.Tensor,
        q_bar_tm1_matrices: torch.Tensor,
        small_epsilon: float,
    ) -> torch.Tensor:
        r"""Compute q(a_{t-1} | a_t, a_0).

        Args:
            one_hot_a0: a one-hot representation of a class type at time step zero, as a tensor with dimension
                [batch_size, number_of_atoms, num_classes]
            one_hot_at: a one-hot representation of a class type at current time step, as a tensor with dimension
                [batch_size, number_of_atoms, num_classes]
             q_matrices: transition matrices at current time step :math:`{Q}_{t}` of dimension
                [batch_size, number_of_atoms, num_classes, num_classes].
            q_bar_matrices: one-shot transition matrices at current time step :math:`\bar{Q}_{t}` of dimension
                [batch_size, number_of_atoms, num_classes, num_classes].
            q_bar_tm1_matrices: one-shot transition matrices at previous time step :math:`\bar{Q}_{t-1}` of dimension
                [batch_size, number_of_atoms, num_classes, num_classes].
            small_epsilon: minimum value for the denominator, to avoid division by zero.

        Returns:
            probabilities over classes,  of dimension [batch_size, num_classes, num_classes]
        """
        q_atm1_given_at_and_0 = get_probability_at_previous_time_step(
            probability_at_zeroth_timestep=one_hot_a0,
            one_hot_probability_at_current_timestep=one_hot_at,
            q_matrices=q_matrices,
            q_bar_matrices=q_bar_matrices,
            q_bar_tm1_matrices=q_bar_tm1_matrices,
            small_epsilon=small_epsilon,
            probability_at_zeroth_timestep_are_normalized=True,
        )
        return q_atm1_given_at_and_0

    @classmethod
    def get_p_atm1_given_at(
        cls,
        predicted_logits: torch.Tensor,
        one_hot_at: torch.Tensor,
        q_matrices: torch.Tensor,
        q_bar_matrices: torch.Tensor,
        q_bar_tm1_matrices: torch.Tensor,
        small_epsilon: float,
    ) -> torch.Tensor:
        r"""Compute p(a_{t-1} | a_t).

        .. math::
            p_\theta(a_{t-1} | a_t) \propto \sum_{\tilde{a}_0} q(a_{t-1}, a_t | \tilde{a}_0)p_\theta(\tilde{a}_0, a_t)

        Args:
            predicted_logits: output of the score network estimating an unnormalized
                :math:`p(a_0 | a_t)` of dimension [batch_size, number_of_atoms, num_type_atoms] where num_type_atoms
                includes the MASK token
            one_hot_at: a one-hot representation of a class type at current time step, as a tensor with dimension
                [batch_size, number_of_atoms, num_classes]
             q_matrices: transition matrices at current time step :math:`{Q}_{t}` of dimension
                [batch_size, number_of_atoms, num_classes, num_classes].
            q_bar_matrices: one-shot transition matrices at current time step :math:`\bar{Q}_{t}` of dimension
                [batch_size, number_of_atoms, num_classes, num_classes].
            q_bar_tm1_matrices: one-shot transition matrices at previous time step :math:`\bar{Q}_{t-1}` of dimension
                [batch_size, number_of_atoms, num_classes, num_classes].
            small_epsilon: minimum value for the denominator, to avoid division by zero.

        Returns:
            one-step transition normalized probabilities of dimension [batch_size, num_classes, num_classes]
        """
        p_atm1_at = get_probability_at_previous_time_step(
            probability_at_zeroth_timestep=predicted_logits,
            one_hot_probability_at_current_timestep=one_hot_at,
            q_matrices=q_matrices,
            q_bar_matrices=q_bar_matrices,
            q_bar_tm1_matrices=q_bar_tm1_matrices,
            small_epsilon=small_epsilon,
            probability_at_zeroth_timestep_are_normalized=False,
        )
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
            predicted_logits: output of the score network logits for :math:`p(a_0 | a_t)`
                of dimension [batch_size, number_of_atoms, num_classes] where num_classes includes the MASK token.
            one_hot_real_atom_types: real atom types :math:`a_0` as one-hot vectors
                of dimension [batch_size, number_of_atoms, num_type_atoms]
            one_hot_noisy_atom_types: noisy atom types :math:`a_t` as one-hot vectors
                of dimension [batch_size, number_of_atoms, num_type_atoms]
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

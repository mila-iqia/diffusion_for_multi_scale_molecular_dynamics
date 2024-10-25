from dataclasses import dataclass
from typing import Any, Dict

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.configuration_parsing import (
    create_parameters_from_configuration_dictionary,
)
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import (
    compute_q_xt_bar_xo,
    compute_q_xt_bar_xtm1,
)


@dataclass(kw_only=True)
class LossParameters:
    """Specific Hyper-parameters for the loss function."""

    coordinates_algorithm: str
    atom_types_ce_weight = 0.001  # default value in gooogle D3PM repo
    atom_types_eps = 1e-8  # avoid divisions by zero
    # https://github.com/google-research/google-research/blob/master/d3pm/images/config.py


@dataclass(kw_only=True)
class MSELossParameters(LossParameters):
    """Specific Hyper-parameters for the MSE loss function."""

    coordinates_algorithm: str = "mse"


@dataclass(kw_only=True)
class WeightedMSELossParameters(LossParameters):
    """Specific Hyper-parameters for the weighted MSE loss function."""

    coordinates_algorithm: str = "weighted_mse"
    # The default values are chosen to lead to a flat loss curve vs. sigma, based on preliminary experiments.
    # These parameters have no effect if the algorithm is 'mse'.
    # The default parameters are chosen such that weights(sigma=0.5) \sim 10^3
    sigma0: float = 0.2
    exponent: float = 23.0259  # ~ 10 ln(10)


class CoordinatesLossCalculator(torch.nn.Module):
    """Class to calculate the loss."""

    def __init__(self, loss_parameters: LossParameters):
        """Init method."""
        super().__init__()
        self.loss_parameters = loss_parameters

    def calculate_unreduced_loss(
        self,
        predicted_normalized_scores: torch.tensor,
        target_normalized_conditional_scores: torch.tensor,
        sigmas: torch.Tensor,
    ) -> torch.tensor:
        """Calculate unreduced Loss.

        All inputs are assumed to be tensors of dimension [batch_size, number_of_atoms, spatial_dimension]. In
        particular, it is assumed that 'sigma' has been broadcast to the same shape as the scores.

        Args:
            predicted_normalized_scores : predicted scores
            target_normalized_conditional_scores : the score targets
            sigmas : the noise

        Returns:
            unreduced_loss: a tensor of shape [batch_size, number_of_atoms, spatial_dimension]. Its mean is the loss.
        """
        raise NotImplementedError


class MSELossCalculator(CoordinatesLossCalculator):
    """Class to calculate the MSE loss."""

    def __init__(self, loss_parameters: MSELossParameters):
        """Init method."""
        super().__init__(loss_parameters)
        self.mse_loss = torch.nn.MSELoss(reduction="none")

    def calculate_unreduced_loss(
        self,
        predicted_normalized_scores: torch.tensor,
        target_normalized_conditional_scores: torch.tensor,
        sigmas: torch.Tensor,
    ) -> torch.tensor:
        """Calculate unreduced Loss.

        All inputs are assumed to be tensors of dimension [batch_size, number_of_atoms, spatial_dimension]. In
        particular, it is assumed that 'sigma' has been broadcast to the same shape as the scores.

        Args:
            predicted_normalized_scores : predicted scores
            target_normalized_conditional_scores : the score targets
            sigmas : the noise

        Returns:
            unreduced_loss: a tensor of shape [batch_size, number_of_atoms, spatial_dimension]. Its mean is the loss.
        """
        assert (
            predicted_normalized_scores.shape
            == target_normalized_conditional_scores.shape
            == sigmas.shape
        ), "Inconsistent shapes"
        unreduced_loss = self.mse_loss(
            predicted_normalized_scores, target_normalized_conditional_scores
        )
        return unreduced_loss


class WeightedMSELossCalculator(MSELossCalculator):
    """Class to calculate the loss."""

    def __init__(self, loss_parameters: WeightedMSELossParameters):
        """Init method."""
        super().__init__(loss_parameters)
        self.register_buffer("sigma0", torch.tensor(loss_parameters.sigma0))
        self.register_buffer("exponent", torch.tensor(loss_parameters.exponent))

    def _exponential_weights(self, sigmas):
        """Compute an exponential weight for the loss."""
        weights = torch.exp(self.exponent * (sigmas - self.sigma0)) + 1.0
        return weights

    def calculate_unreduced_loss(
        self,
        predicted_normalized_scores: torch.tensor,
        target_normalized_conditional_scores: torch.tensor,
        sigmas: torch.Tensor,
    ) -> torch.tensor:
        """Calculate unreduced Loss.

        All inputs are assumed to be tensors of dimension [batch_size, number_of_atoms, spatial_dimension]. In
        particular, it is assumed that 'sigma' has been broadcast to the same shape as the scores.

        Args:
            predicted_normalized_scores : predicted scores
            target_normalized_conditional_scores : the score targets
            sigmas : the noise

        Returns:
            unreduced_loss: a tensor of shape [batch_size, number_of_atoms, spatial_dimension]. It's mean is the loss.
        """
        assert (
            predicted_normalized_scores.shape
            == target_normalized_conditional_scores.shape
            == sigmas.shape
        ), "Inconsistent shapes"

        unreduced_mse_loss = self.mse_loss(
            predicted_normalized_scores, target_normalized_conditional_scores
        )
        weights = self._exponential_weights(sigmas)
        unreduced_loss = unreduced_mse_loss * weights

        return unreduced_loss


class D3PMLossCalculator(torch.nn.Module):
    """Class to calculate the discrete diffusion loss."""

    def __init__(self, loss_parameters: LossParameters):
        """Initialize method."""
        super.__init__()
        # weight of the cross-entropy component
        self.ce_weight = loss_parameters.atom_types_ce_weight
        self.eps = loss_parameters.atom_types_eps

    def kl_loss_term(
        self,
        predicted_unnormalized_probabilities: torch.Tensor,
        one_hot_real_atom_types: torch.Tensor,
        one_hot_noisy_atom_types: torch.Tensor,
        q_matrices: torch.Tensor,
        q_bar_matrices: torch.Tensor,
        q_bar_tm1_matrices: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the KL component of the loss.

        This corresponds to this:

        .. math::

            D_{KL}[q(a_{t-1} | a_t, a_0) || p_theta(a_t | a_{t-1}]

        We are ignoring the t=1 case here as we will use a NLL loss instead.

        Args:
            predicted_unnormalized_probabilities: output of the score network estimating an unnormalized
                :math:`p(a_0 | a_t)` of dimension [batch_size, number_of_atoms, num_type_atoms] where num_type_atoms
                includes the MASK token
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
        q_at_bar_atm1 = compute_q_xt_bar_xtm1(one_hot_noisy_atom_types, q_matrices)
        # dimension of q_at_bar_atm1 : batch_size, number_of_atoms, num_type_atoms
        # q(a_{t-1} | a_0) = a_0 \bar{Q}_{t-1}
        q_atm1_bar_a0 = compute_q_xt_bar_xo(one_hot_real_atom_types, q_bar_tm1_matrices)
        # dimension of q_atm1_bar_a0: batch_size, number_of_atoms, num_type_atoms
        # q(a_t | a_0) = a_0 \bar{Q}_t a_t^T
        q_at_bar_a0 = compute_q_xt_bar_xo(one_hot_real_atom_types, q_bar_matrices)
        q_at_bar_a0 = einops.einsum(
            q_at_bar_a0, one_hot_noisy_atom_types, "... i , ... i -> ..."
        )
        # dimension of q_at_bar_a0: batch_size, number_of_atoms
        posterior_q = (
            q_at_bar_atm1 * q_atm1_bar_a0 / q_at_bar_a0.unsqueeze(-1).clip(min=self.eps)
        )  # clip at eps
        # the unsqueeze in the denominator is to allow a broadcasting
        # posterior q has dimension: batch_size, number_of_atoms, num_type_atoms

        # we now need to compute p_\theta(a_{t-1} | a_t) using
        # p_\theta(a_{t-1} | a_t) \propto \sum_{\tilde{a}_0} q(a_{t-1}, a_t | \tilde{a}_0)p_\theta(\tilde{a}_0, a_t)
        # \propto \sum_{\tilde{a}_0} a_t Q_t^T \circ \tilde{a}_0 \bar{Q}_{t-1} \circ p_\theta(\tilde{a}_0 | a_t)
        # this is equivalent to doing a_t Q_t^T \circ \bar{Q}_{t-1} p_\theta(a_t)
        # with a matrix multiplication in the last step
        # we add a softmax to convert the predictions to normalized probabilities
        p_atpm1_at = q_at_bar_atm1 * einops.einsum(
            q_bar_tm1_matrices,
            torch.nn.softmax(predicted_unnormalized_probabilities, dim=-1),
            "... j i, ... j -> ... i",
        )
        # unit test version TODO
        # p_atm1_at = torch.zeros_like(posterior_q)
        # for i in range(one_hot_real_atom_types.size(-1)):
        #    # a_t Q_t^T is already computed: q_at_bar_atm1
        #    tilde_a_0 = class_index_to_onehot(torch.LongTensor([i]),
        #                                      num_classes=num_classes)  # dimension (1, num_classes)
        #    tilde_a_0_qbar_tm1 = compute_q_xt_bar_xtm1(tilde_a_0, q_bar_tm1_matrices)
        #    p_atm1_at += q_at_bar_atm1 * tilde_a_0_qbar_tm1 * model_predictions[..., i].unsqueeze(-1)

        # get the KL divergence between posterior and predicted prob
        # do not reduce (average) yet as we will replace the samples with t=1 with a NLL loss
        # input of kl_div should be log-probabilities - we add eps to avoid log(0)
        kl_loss = torch.nn.functional.kl_div(
            torch.log(p_atpm1_at + self.eps), posterior_q, reduction="none"
        )
        return kl_loss

    def calculate_unreduced_loss(
        self,
        predicted_unnormalized_probabilities: torch.Tensor,
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

             L_a = E_{a_0 ~ p_data} [ \sum_{t=2}^T E_{at ~ p_{t|0]}[
                    [D_{KL}[q(a_{t-1} | a_t, a_0) || p_theta(a_t | a_{t-1}] - \lambda_CE log p_\theta(a_0 | a_t)]
                    - E_{a1 ~ p_{t=1| 0}} log p_\theta(a_0 | a_1) ]

        Args:
            predicted_unnormalized_probabilities: output of the score network estimating an unnormalized
                :math:`p(a_0 | a_t)` of dimension [batch_size, number_of_atoms, num_type_atoms] where num_type_atoms
                includes the MASK token
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
        # D_{KL}[q(a_{t-1} | a_t, a_0) || p_theta(a_t | a_{t-1}]
        kl_term = self.kl_loss_term(
            predicted_unnormalized_probabilities,
            one_hot_real_atom_types,
            one_hot_noisy_atom_types,
            q_matrices,
            q_bar_matrices,
            q_bar_tm1_matrices,
        )

        # -log p_\theta(a_0 | a_t)
        nll_term = -torch.nn.functional.log_softmax(
            predicted_unnormalized_probabilities
        )

        # if t == 1 (0 for python indexing convention), use the NLL term, otherwise use the KL + \lambda_{CE} NLL
        d3pm_loss = torch.where(
            time_indices.view(-1, 1, 1) == 0,
            nll_term,
            kl_term + self.ce_weight * nll_term,
        )
        return d3pm_loss


class LatticeLoss(torch.nn.Module):
    """Class to calculate the loss for the lattice vectors.

    Placeholder for now.
    """

    def __init__(self):
        super.__init__()

    def calculate_unreduced_loss(self, *args):
        return 0


LOSS_PARAMETERS_BY_ALGO = dict(
    mse=MSELossParameters, weighted_mse=WeightedMSELossParameters
)
LOSS_BY_ALGO = dict(mse=MSELossCalculator, weighted_mse=WeightedMSELossCalculator)


def create_loss_parameters(model_dictionary: Dict[str, Any]) -> LossParameters:
    """Create loss parameters.

    Extract the relevant information from the general configuration dictionary.

    Args:
        model_dictionary : model configuration dictionary.

    Returns:
        loss_parameters: the loss parameters.
    """
    default_dict = dict(algorithm="mse")
    loss_config_dictionary = model_dictionary.get("loss", default_dict)

    loss_parameters = create_parameters_from_configuration_dictionary(
        configuration=loss_config_dictionary,
        identifier="algorithm",
        options=LOSS_PARAMETERS_BY_ALGO,
    )
    return loss_parameters


def create_loss_calculator(loss_parameters: LossParameters) -> AXL:
    """Create Loss Calculator.

    This is a factory method to create the loss calculator.

    Args:
        loss_parameters : parameters defining the loss.

    Returns:
        loss_calculator : the loss calculator for atom types, coordinates, lattice in an AXL namedtuple.
    """
    algorithm = loss_parameters.coordinates_algorithm
    assert (
        algorithm in LOSS_BY_ALGO.keys()
    ), f"Algorithm {algorithm} is not implemented. Possible choices are {LOSS_BY_ALGO.keys()}"

    coordinates_loss = LOSS_BY_ALGO[algorithm](loss_parameters)
    lattice_loss = LatticeLoss  # TODO placeholder
    atom_loss = D3PMLossCalculator(loss_parameters)

    return AXL(
        ATOM_TYPES=atom_loss,
        RELATIVE_COORDINATES=coordinates_loss,
        UNIT_CELL=lattice_loss,
    )

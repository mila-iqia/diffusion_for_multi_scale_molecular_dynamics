import torch

from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import (
    LossParameters, MSELossParameters, WeightedMSELossParameters)


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

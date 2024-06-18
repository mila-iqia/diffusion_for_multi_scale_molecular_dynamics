from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class LossParameters:
    """Specific Hyper-parameters for the loss function."""
    algorithm: str = 'mse'
    # The default values are chosen to lead to a flat loss curve vs. sigma, based on preliminary experiments.
    # These parameters have no effect if the algorithm is 'mse'.
    # The default parameters are chosen such that weights(sigma=0.5) \sim 10^3
    sigma0: float = 0.2
    exponent: float = 23.0259  # ~ 10 ln(10)


class LossCalculator(torch.nn.Module):
    """Class to calculate the loss."""
    def __init__(self, hyperparams: LossParameters):
        """Init method."""
        super().__init__()

        assert hyperparams.algorithm in {'mse', 'weighted_mse'}, f"Unknown loss algorithm '{hyperparams.algorithm}'"

        self.hyperparams = hyperparams

        self.register_buffer("sigma0", torch.tensor(self.hyperparams.sigma0))
        self.register_buffer("exponent", torch.tensor(self.hyperparams.exponent))

        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.compute_weights = hyperparams.algorithm == 'weighted_mse'

    def _exponential_weights(self, sigmas):
        """Compute an exponential weight for the loss."""
        weights = torch.exp(self.exponent * (sigmas - self.sigma0)) + 1.0
        return weights

    def calculate_unreduced_loss(self, predicted_normalized_scores: torch.tensor,
                                 target_normalized_conditional_scores: torch.tensor,
                                 sigmas: torch.Tensor) -> torch.tensor:
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
        assert predicted_normalized_scores.shape == target_normalized_conditional_scores.shape == sigmas.shape, \
            "Inconsistent shapes"

        unreduced_loss = self.mse_loss(predicted_normalized_scores, target_normalized_conditional_scores)
        if self.compute_weights:
            weights = self._exponential_weights(sigmas)
            unreduced_loss = unreduced_loss * weights

        return unreduced_loss

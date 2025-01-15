from dataclasses import dataclass
from typing import Any, AnyStr, Dict

import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork


@dataclass(kw_only=True)
class RegularizerParameters:
    """Base Hyper-parameters for Regularizers."""
    type: str  # what kind of regularizer

    # weighting prefactor for the regularizer loss.
    regularizer_lambda_weight: float = 1.0

    # How many epochs without regularization at the beginning of training
    number_of_burn_in_epochs: int = 0

    def __post_init__(self):
        """Verify conditions in post init."""
        assert self.regularizer_lambda_weight > 0.0, "The regularizer weight must be positive."


class Regularizer:
    """Regularizer.

    This is the base class for Regularizers.

    The goal of a regularizer is to provide auxiliary signal to improve a model's performance.
    A regularizer wouldn't be enough to fully train a model (except in some trivial edge cases),
    and so it must complement a principal source of learning, such as score matching.

    """

    def __init__(self, regularizer_parameters: RegularizerParameters):
        """Init method."""
        self.regularizer_parameters = regularizer_parameters
        self.weight = self.regularizer_parameters.regularizer_lambda_weight
        self.number_of_burn_in_epochs = regularizer_parameters.number_of_burn_in_epochs

    def can_regularizer_run(self):
        """Can regularizer run.

        A convenient method to check if the regularizer can be executed from the global context.
        """
        return True

    def compute_weighted_regularizer_loss(self,
                                          score_network: ScoreNetwork,
                                          augmented_batch: Dict[AnyStr, Any],
                                          current_epoch: int) -> torch.Tensor:
        """Compute Weighted Regularizer Loss.

        This method computes the weighted regularizer loss, using the augmented batch to provide
        anchoring data.

        Args:
            score_network: the score network to be regularized.
            augmented_batch: the augmented batch, which should contain all that is needed to call the score network.
            current_epoch: the current epoch.

        Returns:
            weighted_regularizer_loss: the weighted regularizer loss.
        """
        if current_epoch < self.number_of_burn_in_epochs:
            return torch.tensor(0.0)

        # Validate that the augmented batch is complete.
        score_network._check_batch(augmented_batch)
        loss = self.compute_regularizer_loss(score_network, augmented_batch)

        return self.weight * loss

    def compute_regularizer_loss(self, score_network: ScoreNetwork,
                                 augmented_batch: Dict[AnyStr, Any]) -> torch.Tensor:
        """Compute Regularizer Loss.

        Args:
            score_network: the score network to be regularized.
            augmented_batch: the augmented batch, which should contain all that is needed to call the score network.
            current_epoch: the current epoch.

        Returns:
            regularizer_loss: the regularizer loss.
        """
        raise NotImplementedError("This method must be implemented in a child class.")

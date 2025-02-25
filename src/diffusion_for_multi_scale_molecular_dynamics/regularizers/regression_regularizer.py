from dataclasses import dataclass
from typing import Any, AnyStr, Dict

import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import (
    ScoreNetwork, ScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network_factory import \
    create_score_network
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISY_AXL_COMPOSITION)
from diffusion_for_multi_scale_molecular_dynamics.regularizers.regularizer import (
    Regularizer, RegularizerParameters)


@dataclass(kw_only=True)
class RegressionRegularizerParameters(RegularizerParameters):
    """Parameters for regularization by regression to an analytical score network."""
    type: str = "regression"
    score_network_parameters: ScoreNetworkParameters


class RegressionRegularizer(Regularizer):
    """Regression Regularizer.

    This class implements a regression to a known score network. This makes most sense
    when the target score network is an analytical model.
    """

    def __init__(self, regularizer_parameters: RegressionRegularizerParameters):
        """Init method."""
        super().__init__(regularizer_parameters)
        self.target_score_network = create_score_network(regularizer_parameters.score_network_parameters)

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
        original_noisy_composition = augmented_batch[NOISY_AXL_COMPOSITION]
        original_relative_coordinates = original_noisy_composition.X
        device = original_relative_coordinates.device

        # sample random relative_coordinates
        relative_coordinates = torch.rand(*original_relative_coordinates.shape).to(device)

        # Create the corresponding modified batch
        modified_noisy_composition = AXL(A=original_noisy_composition.A,
                                         X=relative_coordinates,
                                         L=original_noisy_composition.L)
        modified_batch = dict(augmented_batch)
        modified_batch[NOISY_AXL_COMPOSITION] = modified_noisy_composition

        target_normalized_scores = self.target_score_network(modified_batch).X

        normalized_scores = score_network(modified_batch).X

        errors = normalized_scores - target_normalized_scores

        loss = torch.mean((errors)**2)
        return loss

from dataclasses import dataclass
from typing import Any, AnyStr, Dict

import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISY_AXL_COMPOSITION)
from diffusion_for_multi_scale_molecular_dynamics.regularizers.regularizer import (
    Regularizer, RegularizerParameters)


@dataclass(kw_only=True)
class AnalyticalRegressionRegularizerParameters(RegularizerParameters):
    """Parameters for regularization by regression to an analytical score netwowk."""
    type: str = "analytical_regression"
    analytical_score_network_parameters: AnalyticalScoreNetworkParameters


class AnalyticalRegressionRegularizer(Regularizer):
    """Analytical Regression Regularizer.

    This class implements a regression to an analytical score network.

    This is useful mainly for sanity checking, as a real system will not be well described
    by an analytical score network.
    """

    def __init__(self, regularizer_parameters: AnalyticalRegressionRegularizerParameters, device: torch.device):
        """Init method."""
        super().__init__(regularizer_parameters)
        self.analytical_score_network = AnalyticalScoreNetwork(
            regularizer_parameters.analytical_score_network_parameters, device=device
        )

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

        target_normalized_scores = self.analytical_score_network(modified_batch).X

        normalized_scores = score_network(modified_batch).X

        errors = normalized_scores - target_normalized_scores

        loss = torch.mean((errors)**2)
        return loss

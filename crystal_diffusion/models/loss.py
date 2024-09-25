from dataclasses import dataclass
from typing import Any, Dict

import torch

from crystal_diffusion.utils.configuration_parsing import \
    create_parameters_from_configuration_dictionary


@dataclass(kw_only=True)
class LossParameters:
    """Specific Hyper-parameters for the loss function."""
    algorithm: str
    fokker_planck_weight: float = 0.0


@dataclass(kw_only=True)
class MSELossParameters(LossParameters):
    """Specific Hyper-parameters for the MSE loss function."""
    algorithm: str = 'mse'


@dataclass(kw_only=True)
class WeightedMSELossParameters(LossParameters):
    """Specific Hyper-parameters for the weighted MSE loss function."""
    algorithm: str = 'weighted_mse'
    # The default values are chosen to lead to a flat loss curve vs. sigma, based on preliminary experiments.
    # These parameters have no effect if the algorithm is 'mse'.
    # The default parameters are chosen such that weights(sigma=0.5) \sim 10^3
    sigma0: float = 0.2
    exponent: float = 23.0259  # ~ 10 ln(10)


class LossCalculator(torch.nn.Module):
    """Class to calculate the loss."""
    def __init__(self, loss_parameters: LossParameters):
        """Init method."""
        super().__init__()
        self.loss_parameters = loss_parameters

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
            unreduced_loss: a tensor of shape [batch_size, number_of_atoms, spatial_dimension]. Its mean is the loss.
        """
        raise NotImplementedError


class MSELossCalculator(LossCalculator):
    """Class to calculate the MSE loss."""
    def __init__(self, loss_parameters: MSELossParameters):
        """Init method."""
        super().__init__(loss_parameters)
        self.mse_loss = torch.nn.MSELoss(reduction='none')

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
            unreduced_loss: a tensor of shape [batch_size, number_of_atoms, spatial_dimension]. Its mean is the loss.
        """
        assert predicted_normalized_scores.shape == target_normalized_conditional_scores.shape == sigmas.shape, \
            "Inconsistent shapes"
        unreduced_loss = self.mse_loss(predicted_normalized_scores, target_normalized_conditional_scores)
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

        unreduced_mse_loss = self.mse_loss(predicted_normalized_scores, target_normalized_conditional_scores)
        weights = self._exponential_weights(sigmas)
        unreduced_loss = unreduced_mse_loss * weights

        return unreduced_loss


LOSS_PARAMETERS_BY_ALGO = dict(mse=MSELossParameters, weighted_mse=WeightedMSELossParameters)
LOSS_BY_ALGO = dict(mse=MSELossCalculator, weighted_mse=WeightedMSELossCalculator)


def create_loss_parameters(model_dictionary: Dict[str, Any]) -> LossParameters:
    """Create loss parameters.

    Extract the relevant information from the general configuration dictionary.

    Args:
        model_dictionary : model configuration dictionary.

    Returns:
        loss_parameters: the loss parameters.
    """
    default_dict = dict(algorithm='mse')
    loss_config_dictionary = model_dictionary.get("loss", default_dict)

    loss_parameters = (
        create_parameters_from_configuration_dictionary(configuration=loss_config_dictionary,
                                                        identifier="algorithm",
                                                        options=LOSS_PARAMETERS_BY_ALGO))
    return loss_parameters


def create_loss_calculator(loss_parameters: LossParameters) -> LossCalculator:
    """Create Loss Calculator.

    This is a factory method to create the loss calculator.

    Args:
        loss_parameters : parameters defining the loss.

    Returns:
        loss_calculator : the loss calculator.
    """
    algorithm = loss_parameters.algorithm
    assert algorithm in LOSS_BY_ALGO.keys(), \
        f"Algorithm {algorithm} is not implemented. Possible choices are {LOSS_BY_ALGO.keys()}"

    return LOSS_BY_ALGO[algorithm](loss_parameters)

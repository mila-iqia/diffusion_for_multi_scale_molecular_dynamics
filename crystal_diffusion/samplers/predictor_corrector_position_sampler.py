from abc import ABC, abstractmethod

import torch

from crystal_diffusion.samplers.noisy_position_sampler import \
    map_positions_to_unit_cell


class PredictorCorrectorPositionSampler(ABC):
    """This defines the interface for position samplers."""

    def __init__(self, number_of_discretization_steps: int, number_of_corrector_steps: int, **kwargs):
        """Init method."""
        assert number_of_discretization_steps > 0, "The number of discretization steps should be larger than zero"
        assert number_of_corrector_steps >= 0, "The number of corrector steps should be non-negative"

        self.number_of_discretization_steps = number_of_discretization_steps
        self.number_of_corrector_steps = number_of_corrector_steps

    def sample(self, number_of_samples: int) -> torch.Tensor:
        """Sample.

        This method draws a sample using the PR sampler algorithm.

        Args:
            number_of_samples : number of samples to draw.

        Returns:
            position samples: position samples.
        """
        x_ip1 = map_positions_to_unit_cell(self.initialize(number_of_samples))

        for i in range(self.number_of_discretization_steps - 1, -1, -1):
            x_i = map_positions_to_unit_cell(self.predictor_step(x_ip1, i + 1))
            for j in range(self.number_of_corrector_steps):
                x_i = map_positions_to_unit_cell(self.corrector_step(x_i, i))
            x_ip1 = x_i

        return x_i

    @abstractmethod
    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        pass

    @abstractmethod
    def predictor_step(self, x_ip1: torch.Tensor, ip1: int) -> torch.Tensor:
        """Predictor step.

        It is assumed that there are N predictor steps, with index "i" running from N-1 to 0.

        Args:
            x_ip1 : sampled relative positions at step "i + 1".
            ip1 : index "i + 1"

        Returns:
            x_i : sampled relative positions after the predictor step.
        """
        pass

    @abstractmethod
    def corrector_step(self, x_i: torch.Tensor, i: int) -> torch.Tensor:
        """Corrector step.

        It is assumed that there are N predictor steps, with index "i" running from N-1 to 0.
        For each value of "i", there are M corrector steps.
        Args:
            x_i : sampled relative positions at step "i".
            i : index "i" OF THE PREDICTOR STEP.

        Returns:
            x_i_out : sampled relative positions after the corrector step.
        """
        pass

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
            x_i = map_positions_to_unit_cell(self.predictor_step(x_ip1))
            for j in range(self.number_of_corrector_steps):
                x_i = map_positions_to_unit_cell(self.corrector_step(x_i))
            x_ip1 = x_i

        return x_i

    @abstractmethod
    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        pass

    @abstractmethod
    def predictor_step(self, x_ip1: torch.Tensor) -> torch.Tensor:
        """This method must implement a predictor step."""
        pass

    @abstractmethod
    def corrector_step(self, x_i: torch.Tensor) -> torch.Tensor:
        """This method must implement a corrector step."""
        pass

from abc import ABC, abstractmethod

import torch


class PositionGenerator(ABC):
    """This defines the interface for position generators."""

    @abstractmethod
    def sample(self, number_of_samples: int, device: torch.device, unit_cell: torch.Tensor) -> torch.Tensor:
        """Sample.

        This method draws a position sample.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.
            unit_cell: unit cell definition in Angstrom.
                Tensor of dimensions [number_of_samples, spatial_dimension, spatial_dimension]

        Returns:
            samples: relative coordinates samples.
        """
        pass

    @abstractmethod
    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        pass

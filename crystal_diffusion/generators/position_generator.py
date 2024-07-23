from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass(kw_only=True)
class SamplingParameters:
    """Hyper-parameters for diffusion sampling."""
    algorithm: str
    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live.
    number_of_atoms: int  # the number of atoms that must be generated in a sampled configuration.
    number_of_samples: int
    sample_batchsize: Optional[int] = None  # iterate up to number_of_samples with batches of this size
    # if None, use number_of_samples as batchsize
    sample_every_n_epochs: int = 1  # Sampling is expensive; control frequency
    first_sampling_epoch: int = 1  # Epoch at which sampling can begin; no sampling before this epoch.
    cell_dimensions: List[float]  # unit cell dimensions; the unit cell is assumed to be an orthogonal box.
    record_samples: bool = False  # should the predictor and corrector steps be recorded to a file


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

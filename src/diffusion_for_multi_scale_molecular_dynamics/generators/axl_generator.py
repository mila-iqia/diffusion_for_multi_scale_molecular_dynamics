from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@dataclass(kw_only=True)
class SamplingParameters:
    """Hyper-parameters for diffusion sampling."""

    algorithm: str
    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live.
    num_atom_types: int  # number of atom types excluding MASK
    number_of_atoms: (
        int  # the number of atoms that must be generated in a sampled configuration.
    )
    number_of_samples: int
    # iterate up to number_of_samples with batches of this size
    # if None, use number_of_samples as batchsize
    sample_batchsize: Optional[int] = None
    cell_dimensions: List[
        float
    ]  # unit cell dimensions; the unit cell is assumed to be an orthogonal box.  TODO replace with AXL-L
    record_samples: bool = (
        False  # should the predictor and corrector steps be recorded to a file
    )


class AXLGenerator(ABC):
    """This defines the interface for AXL (atom types, reduced coordinates and lattice) generators."""

    @abstractmethod
    def sample(
        self, number_of_samples: int, device: torch.device, unit_cell: torch.Tensor
    ) -> AXL:
        """Sample.

        This method draws a position sample.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.
            unit_cell: unit cell definition in Angstrom.
                Tensor of dimensions [number_of_samples, spatial_dimension, spatial_dimension]

        Returns:
            AXL samples: samples as AXL namedtuple with atom types, reduced coordinates and lattice vectors.
        """
        pass

    @abstractmethod
    def initialize(self, number_of_samples: int) -> AXL:
        """This method must initialize the samples from the fully noised distribution."""
        pass

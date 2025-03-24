import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_unit_cell_to_lattice_parameters


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

    use_fixed_lattice_parameters: bool = False
    cell_dimensions: Optional[List[float]] = None

    record_samples: bool = (
        False  # should the predictor and corrector steps be recorded to a file
    )
    record_samples_corrector_steps: bool = False
    record_atom_type_update: bool = False  # record the information pertaining to generating atom types.

    def __post_init__(self):
        if self.use_fixed_lattice_parameters:
            assert self.cell_dimensions is not None, (
                "If use_fixed_lattice_parameters is True, then cell_dimensions must be provided."
            )
            cell_dimensions = torch.tensor(self.cell_dimensions)
            assert cell_dimensions.dim() == 2, (f"Provided cell_dimensions must be a 2D tensor. "
                                                f"Got {cell_dimensions.shape}.")
            assert cell_dimensions.shape[0] == cell_dimensions.shape[1] == self.spatial_dimension, (
                "The cell_dimensions tensor must have shape [spatial_dimension, spatial_dimension]."
            )
            self.fixed_lattice_parameters = map_unit_cell_to_lattice_parameters(
                cell_dimensions)
        else:
            if not self.use_fixed_lattice_parameters:
                warnings.warn("Using diffusion on lattice parameters. This is experimental and not fully tested.")
            self.fixed_lattice_parameters = None


class AXLGenerator(ABC):
    """This defines the interface for AXL (atom types, reduced coordinates and lattice) generators."""

    @abstractmethod
    def sample(
        self, number_of_samples: int, device: torch.device,
    ) -> AXL:
        """Sample.

        This method draws a configuration sample.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.

        Returns:
            AXL samples: samples as AXL namedtuple with atom types, reduced coordinates and lattice vectors.
        """
        pass

    @abstractmethod
    def initialize(self, number_of_samples: int, device: torch.device) -> AXL:
        """This method must initialize the samples from the fully noised distribution."""
        pass

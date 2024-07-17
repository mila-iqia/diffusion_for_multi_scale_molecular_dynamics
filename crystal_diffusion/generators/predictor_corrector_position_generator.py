import logging
from abc import abstractmethod
from dataclasses import dataclass

import torch
from tqdm import tqdm

from crystal_diffusion.generators.position_generator import (
    PositionGenerator, SamplingParameters)
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class PredictorCorrectorSamplingParameters(SamplingParameters):
    """Hyper-parameters for diffusion sampling with the predictor-corrector algorithm."""
    algorithm: str = 'predictor_corrector'
    number_of_corrector_steps: int = 1


class PredictorCorrectorPositionGenerator(PositionGenerator):
    """This defines the interface for predictor-corrector position generators."""

    def __init__(self, number_of_discretization_steps: int, number_of_corrector_steps: int, spatial_dimension: int,
                 **kwargs):
        """Init method."""
        assert number_of_discretization_steps > 0, "The number of discretization steps should be larger than zero"
        assert number_of_corrector_steps >= 0, "The number of corrector steps should be non-negative"

        self.number_of_discretization_steps = number_of_discretization_steps
        self.number_of_corrector_steps = number_of_corrector_steps
        self.spatial_dimension = spatial_dimension

    def sample(self, number_of_samples: int, device: torch.device, unit_cell: torch.Tensor) -> torch.Tensor:
        """Sample.

        This method draws a sample using the PC sampler algorithm.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.
            unit_cell: unit cell definition in Angstrom.
                Tensor of dimensions [number_of_samples, spatial_dimension, spatial_dimension]

        Returns:
            samples: relative coordinates samples.
        """
        assert unit_cell.size() == (number_of_samples, self.spatial_dimension, self.spatial_dimension), \
            "Unit cell passed to sample should be of size (number of sample, spatial dimension, spatial dimension" \
            + f"Got {unit_cell.size()}"

        x_ip1 = map_relative_coordinates_to_unit_cell(self.initialize(number_of_samples)).to(device)
        forces = torch.zeros_like(x_ip1)

        for i in tqdm(range(self.number_of_discretization_steps - 1, -1, -1)):
            x_i = map_relative_coordinates_to_unit_cell(self.predictor_step(x_ip1, i + 1, unit_cell, forces))
            for _ in range(self.number_of_corrector_steps):
                x_i = map_relative_coordinates_to_unit_cell(self.corrector_step(x_i, i, unit_cell, forces))
            x_ip1 = x_i
        return x_i

    @abstractmethod
    def predictor_step(self, x_ip1: torch.Tensor, ip1: int, unit_cell: torch.Tensor, cartesian_forces: torch.Tensor
                       ) -> torch.Tensor:
        """Predictor step.

        It is assumed that there are N predictor steps, with index "i" running from N-1 to 0.

        Args:
            x_ip1 : sampled relative coordinates at step "i + 1".
            ip1 : index "i + 1"
            unit_cell: sampled unit cell at time step "i + 1".
            cartesian_forces: forces conditioning the diffusion process

        Returns:
            x_i : sampled relative coordinates after the predictor step.
        """
        pass

    @abstractmethod
    def corrector_step(self, x_i: torch.Tensor, i: int, unit_cell: torch.Tensor, cartesian_forces: torch.Tensor
                       ) -> torch.Tensor:
        """Corrector step.

        It is assumed that there are N predictor steps, with index "i" running from N-1 to 0.
        For each value of "i", there are M corrector steps.
        Args:
            x_i : sampled relative coordinates at step "i".
            i : index "i" OF THE PREDICTOR STEP.
            unit_cell: sampled unit cell at time step i.
            cartesian_forces: forces conditioning the diffusion process

        Returns:
            x_i_out : sampled relative coordinates after the corrector step.
        """
        pass

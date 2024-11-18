import logging
from abc import abstractmethod
from dataclasses import dataclass

import torch
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import (
    AXLGenerator, SamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class PredictorCorrectorSamplingParameters(SamplingParameters):
    """Hyper-parameters for diffusion sampling with the predictor-corrector algorithm."""

    algorithm: str = "predictor_corrector"
    number_of_corrector_steps: int = 1
    small_epsilon: float = 1e-8
    one_atom_type_transition_per_step: bool = True
    atom_type_greedy_sampling: bool = True
    atom_type_transition_in_corrector: bool = False


class PredictorCorrectorAXLGenerator(AXLGenerator):
    """Defines the interface for predictor-corrector AXL (atom types, relative coordinates and lattice) generators."""

    def __init__(
        self,
        number_of_discretization_steps: int,
        number_of_corrector_steps: int,
        spatial_dimension: int,
        num_atom_types: int,
        **kwargs,
    ):
        """Init method."""
        assert (
            number_of_discretization_steps > 0
        ), "The number of discretization steps should be larger than zero"
        assert (
            number_of_corrector_steps >= 0
        ), "The number of corrector steps should be non-negative"

        self.number_of_discretization_steps = number_of_discretization_steps
        self.number_of_corrector_steps = number_of_corrector_steps
        self.spatial_dimension = spatial_dimension
        self.num_classes = num_atom_types + 1  # account for the MASK class

    def sample(
        self, number_of_samples: int, device: torch.device, unit_cell: torch.Tensor
    ) -> AXL:
        """Sample.

        This method draws a sample using the PC sampler algorithm.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.
            unit_cell: unit cell definition in Angstrom.  # TODO replace with AXL-L
                Tensor of dimensions [number_of_samples, spatial_dimension, spatial_dimension]

        Returns:
            samples: AXL samples (atom types, relative coordinates, lattice vectors)
        """
        assert unit_cell.size() == (
            number_of_samples,
            self.spatial_dimension,
            self.spatial_dimension,
        ), (
            "Unit cell passed to sample should be of size (number of sample, spatial dimension, spatial dimension"
            + f"Got {unit_cell.size()}"
        )  # TODO replace with AXL-L

        composition_ip1 = self.initialize(number_of_samples, device)

        forces = torch.zeros_like(composition_ip1.X)

        for i in tqdm(range(self.number_of_discretization_steps - 1, -1, -1)):
            composition_i = self.predictor_step(
                composition_ip1, i + 1, unit_cell, forces
            )
            for _ in range(self.number_of_corrector_steps):
                composition_i = self.corrector_step(composition_i, i, unit_cell, forces)
            composition_ip1 = composition_i
        return composition_i

    @abstractmethod
    def predictor_step(
        self,
        composition_ip1: AXL,
        ip1: int,
        unit_cell: torch.Tensor,  # TODO replace with AXL-L
        cartesian_forces: torch.Tensor,
    ) -> AXL:
        """Predictor step.

        It is assumed that there are N predictor steps, with index "i" running from N-1 to 0.

        Args:
            composition_ip1 : sampled AXL composition (atom types, relative coordinates and lattice vectors) at step
                "i + 1".
            ip1 : index "i + 1"
            unit_cell: sampled unit cell at time step "i + 1".  TODO replace with AXL-L
            cartesian_forces: forces conditioning the diffusion process

        Returns:
            composition_i : sampled AXL composition after the predictor step.
        """
        pass

    @abstractmethod
    def corrector_step(
        self,
        composition_i: AXL,
        i: int,
        unit_cell: torch.Tensor,  # TODO replace with AXL-L
        cartesian_forces: torch.Tensor,
    ) -> AXL:
        """Corrector step.

        It is assumed that there are N predictor steps, with index "i" running from N-1 to 0.
        For each value of "i", there are M corrector steps.
        Args:
            composition_i : sampled AXL composition (atom types, relative coordinates and lattice vectors) at step "i".
            i : index "i" OF THE PREDICTOR STEP.
            unit_cell: sampled unit cell at time step i.  # TODO replace with AXL-L
            cartesian_forces: forces conditioning the diffusion process

        Returns:
            corrected_composition_i : sampled composition after the corrector step.
        """
        pass

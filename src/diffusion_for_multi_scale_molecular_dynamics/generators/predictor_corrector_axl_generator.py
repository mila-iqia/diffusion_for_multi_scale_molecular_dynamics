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
        # T = 1 is a dangerous and meaningless edge case.
        assert (
            number_of_discretization_steps > 1
        ), "The number of discretization steps should be larger than one"
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
        starting_noisy_composition = self.initialize(number_of_samples, device)

        composition = self.sample_from_noisy_composition(
            starting_noisy_composition=starting_noisy_composition,
            starting_step_index=self.number_of_discretization_steps,
            ending_step_index=0,
            unit_cell=unit_cell
        )

        return composition

    def sample_from_noisy_composition(
        self, starting_noisy_composition: AXL, starting_step_index: int, ending_step_index: int, unit_cell: torch.Tensor
    ) -> AXL:
        """Sample from noisy composition.

        This method draws a sample using the PC sampler algorithm, starting from a given noisy composition.
        The sampling will start at time index "starting_step_index" and will stop at "ending_step_index" or "0",
        whichever is largest.

        This method is useful to draw intermediate trajectories

        Args:
            starting_noisy_composition: an AXL composition, assumed to correspond to the time step described by
                "starting_step_index".
            starting_step_index: the time index of the starting time for denoising.
            ending_step_index: the time index of the final time for denoising.
            unit_cell: unit cell definition in Angstrom.  # TODO replace with AXL-L
                Tensor of dimensions [number_of_samples, spatial_dimension, spatial_dimension]

        Returns:
            samples: AXL samples (atom types, relative coordinates, lattice vectors) at the time corresponding to
                "ending_step_index".
        """
        assert starting_step_index > ending_step_index, \
            "It is nonsensical for starting_step_index to be smaller or equal to ending_step_index."

        assert starting_step_index > 0, "Starting step should be larger than zero."
        assert ending_step_index >= 0, "ending step should be larger or equal to zero."

        number_of_samples = starting_noisy_composition.X.shape[0]
        assert unit_cell.size() == (
            number_of_samples,
            self.spatial_dimension,
            self.spatial_dimension,
        ), (
            "Unit cell passed to sample should be of size (number of sample, spatial dimension, spatial dimension"
            + f"Got {unit_cell.size()}"
        )  # TODO replace with AXL-L

        composition_ip1 = starting_noisy_composition

        forces = torch.zeros_like(composition_ip1.X)

        for i in tqdm(range(starting_step_index - 1, max(ending_step_index, 0) - 1, -1)):
            # We begin the loop at i = starting_index - 1 because the first predictor step has index "i + 1",
            # such that the first predictor step occurs at = starting_step_index, which is the most natural
            # interpretation of "starting_step_index". The code is more legible with "predictor_step(i+1)" followed
            # by "corrector_step(i)", which is why we do it this way.
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

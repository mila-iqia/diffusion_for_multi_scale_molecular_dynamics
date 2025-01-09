import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import (
    AXLGenerator, SamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.generators.trajectory_initializer import (
    FullRandomTrajectoryInitializer, TrajectoryInitializer,
    TrajectoryInitializerParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_number_of_lattice_parameters

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
        number_of_atoms: int,
        trajectory_initializer: Optional[TrajectoryInitializer] = None,
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
        self.num_lattice_parameters = get_number_of_lattice_parameters(
            spatial_dimension
        )

        if trajectory_initializer is not None:
            self.trajectory_initializer = trajectory_initializer
        else:
            params = TrajectoryInitializerParameters(
                spatial_dimension=spatial_dimension,
                num_atom_types=num_atom_types,
                number_of_atoms=number_of_atoms,
            )
            self.trajectory_initializer = FullRandomTrajectoryInitializer(params)

    def initialize(self, number_of_samples: int, device: torch.device) -> AXL:
        """This method must initialize the samples for sampling trajectory."""
        return self.trajectory_initializer.initialize(number_of_samples, device)

    def sample(
        self,
        number_of_samples: int,
        device: torch.device,
    ) -> AXL:
        """Sample.

        This method draws a sample using the PC sampler algorithm.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.

        Returns:
            samples: AXL samples (atom types, relative coordinates, lattice vectors)
        """
        starting_noisy_composition = self.initialize(number_of_samples, device)

        starting_step_index = self.trajectory_initializer.create_start_time_step_index(
            self.number_of_discretization_steps
        )
        ending_step_index = self.trajectory_initializer.create_end_time_step_index()

        composition = self.sample_from_noisy_composition(
            starting_noisy_composition=starting_noisy_composition,
            starting_step_index=starting_step_index,
            ending_step_index=ending_step_index,
        )

        return composition

    def sample_from_noisy_composition(
        self,
        starting_noisy_composition: AXL,
        starting_step_index: int,
        ending_step_index: int,
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

        Returns:
            samples: AXL samples (atom types, relative coordinates, lattice vectors) at the time corresponding to
                "ending_step_index".
        """
        assert (
            starting_step_index > ending_step_index
        ), "It is nonsensical for starting_step_index to be smaller or equal to ending_step_index."

        assert starting_step_index > 0, "Starting step should be larger than zero."
        assert ending_step_index >= 0, "ending step should be larger or equal to zero."

        composition_ip1 = starting_noisy_composition

        forces = torch.zeros_like(composition_ip1.X)

        for i in tqdm(
            range(starting_step_index - 1, max(ending_step_index, 0) - 1, -1)
        ):
            # We begin the loop at i = starting_index - 1 because the first predictor step has index "i + 1",
            # such that the first predictor step occurs at = starting_step_index, which is the most natural
            # interpretation of "starting_step_index". The code is more legible with "predictor_step(i+1)" followed
            # by "corrector_step(i)", which is why we do it this way.
            composition_i = self.predictor_step(
                composition_ip1, i + 1, forces
            )

            for _ in range(self.number_of_corrector_steps):
                composition_i = self.corrector_step(composition_i, i, forces)
            composition_ip1 = composition_i
        return composition_i

    @abstractmethod
    def predictor_step(
        self,
        composition_ip1: AXL,
        ip1: int,
        cartesian_forces: torch.Tensor,
    ) -> AXL:
        """Predictor step.

        It is assumed that there are N predictor steps, with index "i" running from N-1 to 0.

        Args:
            composition_ip1 : sampled AXL composition (atom types, relative coordinates and lattice vectors) at step
                "i + 1".
            ip1 : index "i + 1"
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
        cartesian_forces: torch.Tensor,
    ) -> AXL:
        """Corrector step.

        It is assumed that there are N predictor steps, with index "i" running from N-1 to 0.
        For each value of "i", there are M corrector steps.
        Args:
            composition_i : sampled AXL composition (atom types, relative coordinates and lattice vectors) at step "i".
            i : index "i" OF THE PREDICTOR STEP.
            cartesian_forces: forces conditioning the diffusion process

        Returns:
            corrected_composition_i : sampled composition after the corrector step.
        """
        pass

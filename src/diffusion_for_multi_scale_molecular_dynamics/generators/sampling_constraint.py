import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@dataclass
class SamplingConstraint:
    """Dataclass to hold the constraints for constrained sampling."""
    total_time_steps: int  # The total number of time steps in denoising

    # the time step index at which free diffusion starts.
    # Should be equal or smaller to total_time_steps
    starting_free_diffusion_time_step: int

    # There should be one AXL reference per time step.
    # Tensors should have shape [time, natoms, ...]
    reference_compositions: AXL

    # Indices of the atoms that should not diffuse freely.
    constrained_atom_indices: torch.Tensor

    def __post_init__(self):
        """Post init method, to validate input."""
        assert self.total_time_steps > 0, "Time steps should be larger than 0."
        assert (
            self.starting_free_diffusion_time_step > 0
        ), "Time steps should be larger than 0."

        assert (
            self.starting_free_diffusion_time_step <= self.total_time_steps
        ), "Starting free diffusion time step should be smaller or equal to the total number of time steps."

        assert (
            type(self.reference_compositions.X) is torch.Tensor
        ), "the reference compositions should torch Tensors."
        assert (
            type(self.reference_compositions.A) is torch.Tensor
        ), "the reference compositions should torch Tensors."
        assert (
            type(self.reference_compositions.L) is torch.Tensor
        ), "the reference compositions should torch Tensors."

        assert len(self.reference_compositions.X.shape) == 3, "X has the wrong shape."
        assert len(self.reference_compositions.A.shape) == 2, "A has the wrong shape."

        number_of_time_steps_, number_of_atoms = self.reference_compositions.A.shape
        assert (
            number_of_time_steps_ == self.total_time_steps
        ), "The number of time steps in the reference compositions should be total_number_of_time_steps."

        number_of_time_steps_, number_of_atoms_, _ = self.reference_compositions.X.shape
        assert (
            number_of_time_steps_ == self.total_time_steps
        ), "The number of time steps in the reference compositions should be total_number_of_time_steps."

        assert (
            number_of_atoms_ == number_of_atoms
        ), "The number of atoms in the reference compositions should be consistent between X and A."

        indices_are_good = torch.logical_and(
            self.constrained_atom_indices >= 0,
            self.constrained_atom_indices < number_of_atoms,
        ).all()

        assert (
            indices_are_good
        ), "the constrained_atom_indices are inconsistent with the reference compositions."


def write_sampling_constraint(
    sampling_constraints: SamplingConstraint, output_path: Path
):
    """Write sampling constraint.

    Args:
        sampling_constraint: object to write to file
        output_path: destination

    Returns:
        None.
    """
    # Write to disk as a dictionary to avoid conflicts if code changes.
    torch.save(dataclasses.asdict(sampling_constraints), output_path)


def read_sampling_constraint(output_path: Path) -> SamplingConstraint:
    """Read Sampling Constraint.

    Args:
        output_path: location of file on disk.

    Returns:
        sampling_constraint: object read from file.
    """
    sampling_constraint_dict = torch.load(output_path)
    return SamplingConstraint(**sampling_constraint_dict)

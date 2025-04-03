import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch


@dataclass
class SamplingConstraint:
    """Dataclass to hold the constraints for constrained sampling."""
    elements: List[str]
    constrained_relative_coordinates: torch.Tensor
    constrained_atom_types: torch.Tensor  # integers assumed to be consistent with the elements list
    constrained_indices: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Post init method, to validate input."""
        number_of_atom_types = len(self.elements)

        assert (
            type(self.constrained_relative_coordinates) is torch.Tensor
        ), "the constrained_relative_coordinates should be a torch Tensor."
        assert (
            self.constrained_relative_coordinates.dtype is torch.float
        ), "the constrained_relative_coordinates should be composed of floats."

        assert (
            len(self.constrained_relative_coordinates.shape) == 2
        ), "constrained_relative_coordinates has the wrong shape."

        assert (
            type(self.constrained_atom_types) is torch.Tensor
        ), "the constrained_atom_types should be a torch Tensor."

        assert (
            self.constrained_atom_types.dtype is torch.long
        ), "the constrained_atom_types should be composed of long integers."

        assert (
            len(self.constrained_atom_types.shape) == 1
        ), "constrained_atom_types has the wrong shape."

        assert (
            self.constrained_relative_coordinates.shape[0]
            == self.constrained_atom_types.shape[0]
        ), "The number of constrained atoms should match"

        assert torch.logical_and(self.constrained_atom_types >= 0,
                                 self.constrained_atom_types < number_of_atom_types).all(), \
            "There is a mismatch between the specified elements and the constrained atom types."

        if self.constrained_indices is not None:
            assert (
                type(self.constrained_indices) is torch.Tensor
            ), "the constrained_indices should be a torch Tensor or None."

            assert (
                len(self.constrained_indices.shape) == 1
            ), "constrained_indices has the wrong shape."

            assert self.constrained_indices.dtype is torch.long, \
                "the constrained_indices, if specified, should be composed of long integers."

            assert (
                self.constrained_relative_coordinates.shape[0]
                == self.constrained_indices.shape[0]
            ), "The number of constrained atoms should match"


def write_sampling_constraint(
    sampling_constraint: SamplingConstraint, output_path: Path
):
    """Write sampling constraint.

    Args:
        sampling_constraint: sampling constraint object to write to file
        output_path: destination

    Returns:
        None.
    """
    # Write to disk as a dictionary to avoid conflicts if code changes.
    torch.save(dataclasses.asdict(sampling_constraint), output_path)


def read_sampling_constraint(output_path: Path) -> SamplingConstraint:
    """Read Sampling Constraint.

    Args:
        output_path: location of file on disk.

    Returns:
        sampling_constraint: object read from file.
    """
    sampling_constraint_dict = torch.load(output_path)
    return SamplingConstraint(**sampling_constraint_dict)

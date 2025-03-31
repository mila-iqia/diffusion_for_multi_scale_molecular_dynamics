import dataclasses
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SamplingConstraint:
    """Dataclass to hold the constraints for constrained sampling."""

    # We use numpy arrays because torch tensors cannot hold strings. We want the "atom types"
    # to be strings to avoid using ill-defined/ error prone integer labels for atom types.
    constrained_relative_coordinates: np.ndarray
    constrained_atom_types: np.ndarray
    constrained_indices: Optional[np.ndarray] = None

    def __post_init__(self):
        """Post init method, to validate input."""
        assert (
            type(self.constrained_relative_coordinates) is np.ndarray
        ), "the constrained_relative_coordinates should be a torch Tensor."
        assert (
            len(self.constrained_relative_coordinates.shape) == 2
        ), "constrained_relative_coordinates has the wrong shape."

        assert (
            type(self.constrained_atom_types) is np.ndarray
        ), "the constrained_atom_types should be a torch Tensor."
        assert (
            len(self.constrained_atom_types.shape) == 1
        ), "constrained_atom_types has the wrong shape."

        assert (
            self.constrained_relative_coordinates.shape[0]
            == self.constrained_atom_types.shape[0]
        ), "The number of constrained atoms should match"

        if self.constrained_indices is not None:
            assert (
                type(self.constrained_indices) is np.ndarray
            ), "the constrained_indices should be a torch Tensor or None."

            assert (
                len(self.constrained_indices.shape) == 1
            ), "constrained_indices has the wrong shape."

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
    with open(output_path, "wb") as fd:
        pickle.dump(dataclasses.asdict(sampling_constraint), fd)


def read_sampling_constraint(output_path: Path) -> SamplingConstraint:
    """Read Sampling Constraint.

    Args:
        output_path: location of file on disk.

    Returns:
        sampling_constraint: object read from file.
    """
    with open(output_path, "rb") as fd:
        sampling_constraint_dict = pickle.load(fd)
    return SamplingConstraint(**sampling_constraint_dict)

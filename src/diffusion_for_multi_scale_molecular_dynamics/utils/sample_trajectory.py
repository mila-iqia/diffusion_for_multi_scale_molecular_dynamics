from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, NamedTuple, Union

import einops
import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


class SampleTrajectory:
    """Sample Trajectory.

    This class aims to record the diffusion sampling process. The goal is to produce
    an artifact that can then be analyzed off-line.
    """

    def __init__(self):
        """Init method."""
        self._internal_data = defaultdict(list)

    def reset(self):
        """Reset data structure."""
        self._internal_data = defaultdict(list)

    def record(self, key: str, entry: Union[Dict[str, Any], NamedTuple]):
        """Record.

        Record data from a trajectory.

        Args:
            key: name of  internal list to which the entry will be added.
            entry: dictionary-like data to be recorded.

        Returns:
            None.
        """
        self._internal_data[key].append(entry)

    def write_to_pickle(self, path_to_pickle: str):
        """Write data to pickle file."""
        with open(path_to_pickle, "wb") as fd:
            torch.save(self._internal_data, fd)


def get_predictor_trajectory(pickle_path: Path) -> AXL:
    """Get predictor trajectory.

    Args:
        pickle_path: location of the output of a sample_trajectory object for a Langevin generator.

    Returns:
        trajectory_axl: trajectory composition object, where each field has dimension [nsamples, time, ...]
    """
    data = torch.load(pickle_path, map_location=torch.device("cpu"))

    predictor_data = data["predictor_step"]

    # The recording might have taken place over multiple batches. Combine corresponding compositions.
    multiple_batch_compositions = defaultdict(list)
    for entry in predictor_data:
        time_index = entry["time_step_index"]
        axl_composition = entry["composition_im1"]
        multiple_batch_compositions[time_index].append(axl_composition)

    list_time_indices = np.sort(np.array(list(multiple_batch_compositions.keys())))[
        ::-1
    ]

    list_compositions = []
    for time_index in list_time_indices:
        batch_compositions = multiple_batch_compositions[time_index]
        composition = AXL(
            A=torch.vstack([c.A for c in batch_compositions]),
            X=torch.vstack([c.X for c in batch_compositions]),
            L=torch.vstack([c.L for c in batch_compositions]),
        )
        list_compositions.append(composition)

    atoms_types = einops.rearrange(
        [c.A for c in list_compositions], "time batch natoms -> batch time natoms"
    )
    relative_coordinates = einops.rearrange(
        [c.X for c in list_compositions],
        "time batch natoms space -> batch time natoms space",
    )
    lattice = einops.rearrange(
        [c.L for c in list_compositions], "time batch d1 d2-> batch time d1 d2"
    )
    trajectory_axl = AXL(A=atoms_types, X=relative_coordinates, L=lattice)

    return trajectory_axl

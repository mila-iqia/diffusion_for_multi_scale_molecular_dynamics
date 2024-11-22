from collections import defaultdict
from typing import Any, Dict, NamedTuple, Union

import torch


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

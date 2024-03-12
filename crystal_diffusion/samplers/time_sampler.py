from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class TimeParameters:
    """Time sampling parameters."""
    total_time_steps: int
    random_seed: int = 1234


class TimeSampler:
    """Time Sampler.

    This class will produce time step samples as needed.
    The times will be normalized to be between 0 and 1.
    """
    def __init__(self, time_parameters: TimeParameters):
        """Init method.

        Args:
            time_parameters: parameters needed to instantiate the time sampler.
        """
        self.time_parameters = time_parameters
        self.time_step_array = torch.linspace(0, 1, time_parameters.total_time_steps)

        self._maximum_index = time_parameters.total_time_steps - 1
        self._minimum_index = 1  # we don't want to sample "0".
        self._rng = torch.manual_seed(time_parameters.random_seed)

    def get_random_time_step_indices(self, shape: Tuple[int]) -> torch.Tensor:
        """Random time step indices.

        Generate random indices that correspond to valid time steps.
        This sampling avoids index "0", which corresponds to time "0".

        Args:
            shape: shape of the random index array.

        Returns:
            time_step_indices: random time step indices in a tensor of shape "shape".
        """
        random_indices = torch.randint(self._minimum_index,
                                       self._maximum_index + 1,  # +1 because the maximum value is not sampled
                                       size=shape,
                                       generator=self._rng)
        return random_indices

    def get_time_steps(self, indices: torch.Tensor) -> torch.Tensor:
        """Get time steps.

        Extract the time steps from the internal data structure, given desired indices.

        Args:
            indices: the time step indices. Must be between 0 and "total_number_of_time_steps" - 1.

        Returns:
            time_steps: random time step indices in a tensor of shape "shape".
        """
        assert torch.all(indices >= 0), "indices must be non-negative"
        assert torch.all(indices <= self._maximum_index), \
            f"indices must be smaller than or equal to {self._maximum_index}"
        return self.time_step_array.take(indices)

    def get_forward_iterator(self):
        """Get forward iterator.

        Iterate over time steps for indices in 0,..., T-1, where T is the maximum index.
        This is useful for the NOISING process, which is not formally needed in the formalism,
        but might be useful for sanity checking / debugging.

        Returns:
            forward_iterator: an iterator over (index, time_step) that iterates over increasing values.
        """
        return iter(enumerate(self.time_step_array[:-1]))

    def get_backward_iterator(self):
        """Get backward iterator.

        Iterate over time steps for indices in T,...,1 , where T is the maximum index.
        This is useful for the DENOISING process.

        Returns:
            backward_iterator: an iterator over (index, time_step) that iterates over decreasing values.
        """
        maximum_index = len(self.time_step_array) - 1
        indices = range(maximum_index, 0, -1)
        return zip(indices, self.time_step_array.flip(dims=(0,)))

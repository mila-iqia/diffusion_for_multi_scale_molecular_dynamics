"""The result state of a dynamic-driver (ARTn or MD) run."""

from enum import Enum, auto


class CalculationState(Enum):
    """State of a dynamic-driver (ARTn or MD) calculation."""

    SUCCESS = auto()       # the run finished without the uncertainty exceeding the threshold.
    INTERRUPTION = auto()  # the run halted early: an uncertain structure was found.
    ERROR = auto()         # the run failed or produced unusable output.

from enum import Enum


class CalculationState(Enum):
    """State of an ARTn calculation."""
    SUCCESS = True
    INTERRUPTION = False

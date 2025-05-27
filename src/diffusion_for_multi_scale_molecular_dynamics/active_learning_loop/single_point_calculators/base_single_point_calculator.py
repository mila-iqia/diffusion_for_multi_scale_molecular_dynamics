from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pymatgen.core import Structure


@dataclass(kw_only=True)
class SinglePointCalculation:
    """A data structure to hold the output of a single point calculator."""
    calculation_type: str
    structure: Structure
    forces: np.ndarray
    energy: float
    uncertainties: Optional[np.ndarray] = None


class BaseSinglePointCalculator:
    """Base Single Point Calculator.

    This base class defines the interface for performing "single-point" MLIP calculations.
    Here, "single-point" means a single structure, as opposed to, say, a trajectory.
    """
    def __init__(self, args, **kwargs):
        """Init method."""
        pass

    @abstractmethod
    def calculate(self, structure: Structure) -> SinglePointCalculation:
        """This method just defines the API."""
        raise NotImplementedError("This method must be implemented in a child class.")

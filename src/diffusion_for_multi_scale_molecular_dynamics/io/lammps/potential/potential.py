"""Base class for interatomic potentials expressed as LAMMPS commands."""

from abc import ABC, abstractmethod
from typing import List, Optional


class LammpsPotential(ABC):
    """A potential that writes its own interaction section into a LAMMPS input."""

    calculation_type: str = "lammps"
    name: str = "LAMMPS"  # human-readable potential name, used in logs; overridden by each potential.

    @abstractmethod
    def interaction_commands(self, elements_string: str, with_uncertainty: bool = False) -> List[str]:
        """Return the interaction section: pair_style, pair_coeff and any uncertainty setup.

        This can run with or without uncertainty.
        """
        raise NotImplementedError("must be implemented in a child class.")

    def dump_fields(self, with_uncertainty: bool = False) -> List[str]:
        """Return the per-atom fields written to the main dump."""
        return ["id", "element", "x", "y", "z", "fx", "fy", "fz"]

    def uncertainty_field(self) -> Optional[str]:
        """Return the per-atom uncertainty column name, or None if the potential has no uncertainty."""
        return None

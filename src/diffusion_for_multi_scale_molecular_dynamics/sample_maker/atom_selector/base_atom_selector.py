from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(kw_only=True)
class BaseAtomSelectorParameters:
    """Parameters controlling the atom selection."""
    algorithm: str


class BaseAtomSelector(ABC):
    """Base class for atom selection."""

    def __init__(self, atom_selector_parameters: BaseAtomSelectorParameters):
        """Init method."""
        self.atom_selector_parameters = atom_selector_parameters

    @abstractmethod
    def select_central_atoms(self, uncertainty_per_atom: np.array) -> np.array:
        """Select the central atoms to define the problematic environments.

        Args:
            uncertainty_per_atom: value for all atoms

        Returns:
            indices of all selected atoms, sorted from the atom with the highest uncertainty to the lowest.
        """
        pass

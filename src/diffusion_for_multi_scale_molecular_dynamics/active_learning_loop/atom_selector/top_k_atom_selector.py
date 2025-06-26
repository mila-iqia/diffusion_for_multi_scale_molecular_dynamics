from dataclasses import dataclass

import numpy as np

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.base_atom_selector import (
    BaseAtomSelector, BaseAtomSelectorParameters)


@dataclass(kw_only=True)
class TopKAtomSelectorParameters(BaseAtomSelectorParameters):
    """Parameters controlling the atom selection."""
    algorithm: str = "top_k"
    top_k_environment: int

    def __post_init__(self):
        """Post init."""
        assert self.top_k_environment > 0, \
            f"top_k_environment should be positive. Got {self.top_k_environment}"


class TopKAtomSelector(BaseAtomSelector):
    """Base class for atomic environment excision."""

    def __init__(self, atom_selector_parameters: TopKAtomSelectorParameters):
        """Init method."""
        super().__init__(atom_selector_parameters)
        self.atom_selector_parameters = atom_selector_parameters
        self.top_k = atom_selector_parameters.top_k_environment

    def select_central_atoms(self, uncertainty_per_atom: np.array) -> np.array:
        """Find the top k atoms with the highest uncertainty values.

        Args:
            uncertainty_per_atom: uncertainty value for all atoms

        Returns:
            top_k_indices_descending: indices of all atoms with a value over the threshold, sorted so the first index
                has the highest uncertainty
        """
        sorted_indices = np.argsort(uncertainty_per_atom)
        # Take the last k indices, which correspond to the k largest values
        top_k_indices = sorted_indices[-self.top_k:]
        # Reverse the order to have the indices corresponding to the highest values first
        top_k_indices_descending = top_k_indices[::-1]
        return top_k_indices_descending

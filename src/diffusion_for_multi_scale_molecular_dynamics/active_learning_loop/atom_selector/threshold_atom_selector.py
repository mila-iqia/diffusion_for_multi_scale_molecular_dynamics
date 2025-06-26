from dataclasses import dataclass

import numpy as np

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.base_atom_selector import (
    BaseAtomSelector, BaseAtomSelectorParameters)


@dataclass(kw_only=True)
class ThresholdAtomSelectorParameters(BaseAtomSelectorParameters):
    """Parameters controlling the atom selection."""
    algorithm: str = "threshold"
    uncertainty_threshold: float

    def __post_init__(self):
        """Post init."""
        assert self.uncertainty_threshold > 0., "Only positive uncertainty thresholds are allowed."


class ThresholdAtomSelector(BaseAtomSelector):
    """Atom selection based on an uncertainty threshold."""

    def __init__(self, atom_selector_parameters: ThresholdAtomSelectorParameters):
        """Init method."""
        super().__init__(atom_selector_parameters)
        self.atom_selection_threshold = atom_selector_parameters.uncertainty_threshold

    def select_central_atoms(self, uncertainty_per_atom: np.array) -> np.array:
        """Find all atoms with an uncertainty value above the specified threshold.

        Args:
            uncertainty_per_atom: uncertainty value for all atoms

        Returns:
            sorted_indices: indices of all atoms with a value over the threshold, sorted so the first index has the
                highest uncertainty
        """
        # using np.where returns a tuple with the first element as the relevant indices
        atom_over_threshold_indices = np.where(
            uncertainty_per_atom > self.atom_selection_threshold
        )[0]
        uncertainty_values = uncertainty_per_atom[atom_over_threshold_indices]
        # reorder the indices so the first element in that list
        sorted_indices = atom_over_threshold_indices[np.argsort(uncertainty_values)][
            ::-1
        ]
        return sorted_indices

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@dataclass(kw_only=True)
class BaseEnvironmentExcisionArguments:
    """Parameters controlling the environment excision."""

    algorithm: str
    uncertainty_threshold: Optional[float] = (
        None  # excise the environment for all atoms with an uncertainty higher than this value
    )
    excise_top_k_environment: Optional[int] = (
        None  # if set, excise the top k environments with the highest uncertainty values.
    )

    def __post_init__(self):
        assert (
            self.uncertainty_threshold is not None
            or self.excise_top_k_environment is not None
        ), "uncertainty_threshold or excise_top_k_environment should be defined"

        if self.uncertainty_threshold is not None:
            assert (
                self.excise_top_k_environment is None
            ), "Only one of uncertainty_threshold and excise_top_k_environment should be defined."

        if self.excise_top_k_environment is not None:
            assert (
                self.excise_top_k_environment > 0
            ), f"excise_top_k_environment should be positive. Got {self.excise_top_k_environment}"


class BaseEnvironmentExcision(ABC):
    def __init__(self, excision_arguments: BaseEnvironmentExcisionArguments):
        self.arguments = excision_arguments
        if excision_arguments.uncertainty_threshold is None:
            self.atom_selection_method = "topk"
            self.atom_selection_topk = excision_arguments.excise_top_k_environment
        else:
            self.atom_selection_method = "threshold"
            self.atom_selection_threshold = excision_arguments.uncertainty_threshold

    def select_central_atoms(self, uncertainty_per_atom: np.array) -> np.array:
        """Select the central atoms to define the problematic environments.

        Args:
            uncertainty_per_atom: value for all atoms

        Returns:
            indices of all selected atoms, sorted from the atom with the highest uncertainty to the lowest.
        """
        if self.atom_selection_method == "topk":
            return self._select_topk_atoms(uncertainty_per_atom)
        else:  # self.atom_selection_method == "threshold":
            return self._select_threshold_atoms(uncertainty_per_atom)

    def _select_topk_atoms(self, uncertainty_per_atom: np.array) -> np.array:
        """Find the top k atoms with the highest uncertainty values.

        Args:
            uncertainty_per_atom: uncertainty value for all atoms

        Returns:
            top_k_indices_descending: indices of all atoms with a value over the threshold, sorted so the first index
                has the highest uncertainty
        """
        sorted_indices = np.argsort(uncertainty_per_atom)
        # Take the last k indices, which correspond to the k largest values
        top_k_indices = sorted_indices[-self.atom_selection_topk:]
        # Reverse the order to have the indices corresponding to the highest values first
        top_k_indices_descending = top_k_indices[::-1]
        return top_k_indices_descending

    def _select_threshold_atoms(self, uncertainty_per_atom: np.array) -> np.array:
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

    def excise_environments(
        self, structure: AXL, uncertainty_per_atom: np.array
    ) -> List[AXL]:
        """Extract all environments around the atoms satisfying the uncertainty constraints.

        This calls the method _excise_one_environment for each atom with a high enough uncertainty.

        Args:
            structure: crystal structure, including atomic species, coordinates and lattice parameters
            uncertainty_per_atom: uncertainty associated to each atom. The order is assumed to be the same as those in
                the structure variable.

        Returns:
            environments: list of excised spheres around the highest uncertainties atoms as a list of AXL.
        """
        central_atoms = self.select_central_atoms(uncertainty_per_atom)
        environments = [
            self._excise_one_environment(structure, atom) for atom in central_atoms
        ]
        return environments

    @abstractmethod
    def _excise_one_environment(self, structure: AXL, central_atom_idx: int) -> AXL:
        """Excise the relevant atomic environment around the central atom.

        Args:
            structure: complete structure to excise from
            central_atom_idx: atom at the center of the excised region

        Returns:
            excised_substructure: all atoms within a distance radial_cutoff of the central atom (including itself).
        """
        pass


@dataclass(kw_only=True)
class NoOpEnvironmentExcisionArguments(BaseEnvironmentExcisionArguments):
    algorithm = "NoOpExcision"


class NoOpEnvironmentExcision(BaseEnvironmentExcision):
    """Trivial environment excision method that returns the full environment without modifications."""

    def _excise_one_environment(
        self,
        structure: AXL,
        central_atom_idx: int,
    ) -> AXL:
        return structure

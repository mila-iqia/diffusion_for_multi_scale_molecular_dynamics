from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

CALCULATION_TYPE_KEY = "calculation_type"
ACTIVE_ENVIRONMENT_INDICES_KEY = "active_environment_indices"


@dataclass(kw_only=True)
class SinglePointCalculation:
    """A data structure to hold the output of a single point calculator."""

    calculation_type: str
    atoms: Atoms
    forces: np.ndarray
    energy: float
    uncertainties: Optional[np.ndarray] = None
    additional_information: Optional[Dict[str, Any]] = None

    def to_atoms(self, active_environment_indices: Optional[List[int]] = None) -> Atoms:
        """Return an ase.Atoms carrying the energy/forces (on a calculator) and the metadata in info.

        The training database stores only ase.Atoms, so the calculation type and the optional FLARE
        active-environment indices travel in ``atoms.info``.
        """
        atoms = self.atoms
        atoms.calc = SinglePointCalculator(
            atoms, energy=float(self.energy), forces=np.asarray(self.forces, dtype=float)
        )
        atoms.info[CALCULATION_TYPE_KEY] = self.calculation_type
        if active_environment_indices is not None:
            atoms.info[ACTIVE_ENVIRONMENT_INDICES_KEY] = np.asarray(active_environment_indices, dtype=int)
        return atoms

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> "SinglePointCalculation":
        """Rebuild a SinglePointCalculation from a labelled ase.Atoms (energy + forces on its calculator)."""
        return cls(
            calculation_type=atoms.info.get(CALCULATION_TYPE_KEY, "labelled"),
            atoms=atoms,
            forces=np.asarray(atoms.get_forces(), dtype=float),
            energy=float(atoms.get_potential_energy()),
        )


def get_active_environment_indices(atoms: Atoms) -> Optional[List[int]]:
    """Return the active-environment indices stored on a labelled atoms, or None if none were recorded."""
    indices = atoms.info.get(ACTIVE_ENVIRONMENT_INDICES_KEY)
    return None if indices is None else [int(index) for index in indices]


class BaseSinglePointCalculator:
    """Base Single Point Calculator.

    This base class defines the interface for performing "single-point" MLIP calculations.
    Here, "single-point" means a single structure, as opposed to, say, a trajectory.
    """

    name = "the oracle"  # human-readable oracle name, used in logs; overridden by each calculator.

    def __init__(self, args, **kwargs):
        """Init method."""
        pass

    @abstractmethod
    def calculate(
        self, atoms: Atoms, results_path: Optional[Path] = None
    ) -> SinglePointCalculation:
        """This method just defines the API."""
        raise NotImplementedError("This method must be implemented in a child class.")

    def calculate_many(self, list_atoms: List[Atoms]) -> List[SinglePointCalculation]:
        """Calculate several configurations at once.

        The default loops over ``calculate``; subclasses may override it with a faster batched
        implementation (e.g. a single looping LAMMPS input).
        """
        return [self.calculate(atoms) for atoms in list_atoms]

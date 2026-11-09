"""Base class for the trainable core of an MLIP."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential
from diffusion_for_multi_scale_molecular_dynamics.io.training_database import \
    TrainingDatabase
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    SinglePointCalculation


class BaseMLIPTrainer(ABC):
    """The trainable core of an MLIP: it learns from labelled structures and exports a LAMMPS potential."""

    def __init__(self, training_database: Optional[TrainingDatabase] = None):
        """Init method.

        Args:
            training_database: the training set (single source of truth). May be attached later (e.g. by the
                active learning loop) via ``set_training_database``.
        """
        self._training_database = training_database

    @property
    def training_database(self) -> Optional[TrainingDatabase]:
        """The training database backing this trainer."""
        return self._training_database

    def set_training_database(self, training_database: TrainingDatabase) -> None:
        """Attach (or replace) the training database backing this trainer."""
        self._training_database = training_database

    @property
    def labelled_calculations(self) -> List[SinglePointCalculation]:
        """The training set, read from the training database (empty if none is attached).

        The trainer does not remember what it was fed: the database is the single source of truth.
        """
        if self._training_database is None:
            return []
        return [SinglePointCalculation.from_atoms(atoms) for atoms in self._training_database.labelled_atoms]

    def add_labelled_structure(
        self, single_point_calculation: SinglePointCalculation, active_environment_indices: List[int]
    ) -> None:
        """Fold a labelled structure into the model. Persistence is the training database's job, not the trainer's."""
        self._add_labelled_structure_to_model(single_point_calculation, active_environment_indices)

    @abstractmethod
    def _add_labelled_structure_to_model(
        self, single_point_calculation: SinglePointCalculation, active_environment_indices: List[int]
    ) -> None:
        """Fold a labelled structure into the underlying model."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def fit(self) -> None:
        """Fit the model on the current training set."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def write_checkpoint(self, output_directory: Path) -> LammpsPotential:
        """Write the model into output_directory (checkpoint + LAMMPS files) and return the deployed potential."""
        raise NotImplementedError("must be implemented in a child class.")

    @classmethod
    @abstractmethod
    def load_checkpoint(cls, checkpoint_path: Path) -> "BaseMLIPTrainer":
        """Reconstruct the trainer from a checkpoint."""
        raise NotImplementedError("must be implemented in a child class.")

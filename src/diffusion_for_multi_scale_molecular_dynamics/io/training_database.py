"""Filesystem-backed training database for the active learning loop.

The database is rooted directly at the campaign working directory: per-epoch folders plus the
provided/training .traj files. It is the single source of truth for the training set and the bookkeeping
that drives crash recovery. It deals only in ase.Atoms (never SinglePointCalculation), keeping ``io`` free
of any ``oracle`` dependency; energies/forces ride on an attached calculator and indices/uncertainty live
in ``atoms.info``.

The training set is exactly ``precomputation/first_training.traj`` plus every committed
``epoch_N/oracle.traj``; no other trajectory in the folder is considered.

Layout (the working directory itself)::

    provided_configurations.traj    the PROVIDED starting configuration(s); the augmentation seed. NOT part
                                     of the training set.
    precomputation/                 the precomputation model (fit before the first round) and its data:
      first_training.traj           the labelled first training set (provided + augmented) for precomputation.
      potential...                  the deployed precomputation model files.
    epoch_1/
      dynamic/       dynamic driver working directory
      dynamic.traj   stage 1 commit: the uncertain configuration (+ per-atom 'uncertainty')
      generate.traj  stage 2 commit: the generated samples (+ per-atom 'active_indices'/'constrained_atom_indices')
      oracle/        oracle working directory
      oracle.traj    stage 3 commit: the labelled configurations (energy + forces)
      model/         stage 4 commit: the deployed MLIP checkpoint
    epoch_2/ ...
"""

import shutil
from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple

from diffusion_for_multi_scale_molecular_dynamics.io.utils import (
    read_atoms_trajectory, write_atoms_trajectory)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    FIRST_TRAINING_FILENAME, PRECOMPUTATION_DIRECTORY_NAME,
    PROVIDED_CONFIGURATIONS_FILENAME)


class Stage(Enum):
    """The first step of a round still left to run; where a (re)start re-enters the loop.

    A round always runs DRIVER -> GENERATE -> ORACLE -> TRAIN, so resuming at a stage means the earlier
    stages are already done on disk and only this stage onward needs to run.
    """

    DRIVER = auto()
    GENERATE = auto()
    ORACLE = auto()
    TRAIN = auto()


class TrainingDatabase:
    """A ``database/`` directory holding the training set and the per-epoch staged artifacts."""

    def __init__(self, database_directory: Path):
        """Init method.

        Args:
            database_directory: the ``database/`` root; created if it does not exist.
        """
        self._root = Path(database_directory)
        self._root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_scratch(cls, working_directory: Path) -> "TrainingDatabase":
        """Create the training database of a fresh campaign (rooted directly at the working directory)."""
        return cls(Path(working_directory))

    @classmethod
    def from_computation_folder(cls, working_directory: Path) -> "TrainingDatabase":
        """Open the training database of an existing campaign (rooted directly at the working directory)."""
        return cls(Path(working_directory))

    @classmethod
    def get_epoch_and_stage(cls, working_directory: Path, restart_from_stage: str = "auto") -> Tuple[int, "Stage"]:
        """Scan a working directory and return the (epoch, stage) to (re)enter at (0 = precomputation)."""
        return cls.from_computation_folder(working_directory).resume_point(restart_from_stage)

    # ------------------------------------------------------------------ paths
    def epoch_directory(self, epoch: int) -> Path:
        """The ``epoch_{epoch}`` folder (created on demand)."""
        directory = self._root / f"epoch_{epoch}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def dynamic_directory(self, epoch: int) -> Path:
        """The dynamic driver working directory for an epoch (created on demand)."""
        directory = self.epoch_directory(epoch) / "dynamic"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def oracle_directory(self, epoch: int) -> Path:
        """The oracle working directory for an epoch (created on demand)."""
        directory = self.epoch_directory(epoch) / "oracle"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def model_directory(self, epoch: int) -> Path:
        """The committed-model directory: precomputation for epoch 0, ``epoch_N/model`` otherwise."""
        if epoch == 0:
            return self.precomputation_model_directory()
        directory = self.epoch_directory(epoch) / "model"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _dynamic_trajectory_path(self, epoch: int) -> Path:
        return self._root / f"epoch_{epoch}" / "dynamic.traj"

    def _generate_trajectory_path(self, epoch: int) -> Path:
        return self._root / f"epoch_{epoch}" / "generate.traj"

    def _oracle_trajectory_path(self, epoch: int) -> Path:
        return self._root / f"epoch_{epoch}" / "oracle.traj"

    # -------------------------------------------------------------- discovery
    def _epoch_numbers(self) -> List[int]:
        """The existing epoch numbers, sorted ascending."""
        numbers = []
        for path in self._root.glob("epoch_*"):
            if path.is_dir():
                suffix = path.name[len("epoch_"):]
                if suffix.isdigit():
                    numbers.append(int(suffix))
        return sorted(numbers)

    @property
    def epoch(self) -> int:
        """The highest existing epoch number, or 0 if there are no epochs yet."""
        numbers = self._epoch_numbers()
        return numbers[-1] if numbers else 0

    # ------------------------------------------------------------ precomputation
    def provided_configurations_path(self) -> Path:
        """Path to the provided starting configuration(s) trajectory (the augmentation seed)."""
        return self._root / PROVIDED_CONFIGURATIONS_FILENAME

    def has_provided_configurations(self) -> bool:
        """Whether a provided starting-configuration trajectory exists."""
        return self.provided_configurations_path().is_file()

    def write_provided_configurations(self, configurations: List) -> Path:
        """Write the provided starting configuration(s) to provided_configurations.traj."""
        return write_atoms_trajectory(list(configurations), self.provided_configurations_path())

    def read_provided_configurations(self) -> List:
        """Read back the provided starting configuration(s)."""
        return read_atoms_trajectory(self.provided_configurations_path())

    def training_configurations_path(self) -> Path:
        """Path to the labelled first-training trajectory (the precomputation training set)."""
        return self._root / PRECOMPUTATION_DIRECTORY_NAME / FIRST_TRAINING_FILENAME

    def append_training_configurations(self, labelled_configurations: List) -> Path:
        """Append labelled configurations to precomputation/first_training.traj."""
        path = self.training_configurations_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = read_atoms_trajectory(path) if path.is_file() else []
        return write_atoms_trajectory(existing + list(labelled_configurations), path)

    def number_of_labelled_environments(self) -> int:
        """Total number of atomic environments (atoms) across the labelled training set."""
        return sum(len(atoms) for atoms in self.labelled_atoms)

    def precomputation_model_directory(self) -> Path:
        """The directory holding the precomputation model, fit before the first round (created on demand)."""
        directory = self._root / PRECOMPUTATION_DIRECTORY_NAME
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def is_precomputation_model_committed(self) -> bool:
        """Whether the precomputation model has been written (a fit finished).

        The precomputation model files share the folder with ``first_training.traj``, so the training
        trajectory alone does not count as a committed model.
        """
        directory = self._root / PRECOMPUTATION_DIRECTORY_NAME
        if not directory.is_dir():
            return False
        return any(path.suffix != ".traj" for path in directory.iterdir())

    # ---------------------------------------------------------------- staging
    def is_dynamic_committed(self, epoch: int) -> bool:
        """Whether the epoch's dynamic (stage 1) artifact has been written."""
        return self._dynamic_trajectory_path(epoch).is_file()

    def is_generate_committed(self, epoch: int) -> bool:
        """Whether the epoch's generate (stage 2) artifact has been written."""
        return self._generate_trajectory_path(epoch).is_file()

    def is_oracle_committed(self, epoch: int) -> bool:
        """Whether the epoch's oracle (stage 3) artifact has been written."""
        return self._oracle_trajectory_path(epoch).is_file()

    def is_model_committed(self, epoch: int) -> bool:
        """Whether the epoch's model is committed (epoch 0 is the precomputation model)."""
        if epoch == 0:
            return self.is_precomputation_model_committed()
        model_directory = self._root / f"epoch_{epoch}" / "model"
        return model_directory.is_dir() and any(model_directory.iterdir())

    def write_dynamic(self, epoch: int, uncertain_configuration) -> Path:
        """Write the epoch's uncertain configuration (stage 1) to ``dynamic.traj``."""
        return write_atoms_trajectory([uncertain_configuration], self._dynamic_trajectory_path(epoch))

    def write_generate(self, epoch: int, generated_samples: List) -> Path:
        """Write the epoch's generated samples (stage 2) to ``generate.traj``."""
        return write_atoms_trajectory(generated_samples, self._generate_trajectory_path(epoch))

    def write_oracle(self, epoch: int, labelled_configurations: List) -> Path:
        """Write the epoch's labelled configurations (stage 3) to ``oracle.traj``."""
        return write_atoms_trajectory(labelled_configurations, self._oracle_trajectory_path(epoch))

    def read_dynamic(self, epoch: int):
        """Read back the epoch's uncertain configuration."""
        return read_atoms_trajectory(self._dynamic_trajectory_path(epoch))[0]

    def read_generate(self, epoch: int) -> List:
        """Read back the epoch's generated samples."""
        return read_atoms_trajectory(self._generate_trajectory_path(epoch))

    def read_oracle(self, epoch: int) -> List:
        """Read back the epoch's labelled configurations."""
        return read_atoms_trajectory(self._oracle_trajectory_path(epoch))

    # ----------------------------------------------------------- training set
    def training_trajectory_paths(self) -> List[Path]:
        """The trajectory files that make up the training set, in order.

        This is ``precomputation/first_training.traj`` (when present) followed by every committed
        ``epoch_N/oracle.traj``; no other trajectory in the folder is considered. The dynamic (uncertain)
        configurations and the provided_configurations seed are deliberately excluded.
        """
        trajectory_paths = []
        if self.training_configurations_path().is_file():
            trajectory_paths.append(self.training_configurations_path())
        trajectory_paths += [
            self._oracle_trajectory_path(epoch)
            for epoch in self._epoch_numbers()
            if self.is_oracle_committed(epoch)
        ]
        return trajectory_paths

    @property
    def labelled_atoms(self) -> List:
        """The full training set: the atoms read from every ``training_trajectory_paths`` file."""
        atoms_list = []
        for trajectory_path in self.training_trajectory_paths():
            atoms_list.extend(read_atoms_trajectory(trajectory_path))
        return atoms_list

    def check_labelled_atoms_have_energy_and_forces(self) -> None:
        """Verify every training frame carries an energy and forces (call once, e.g. on restart).

        This only checks the attached calculator's stored results, so it never triggers an actual
        energy/force evaluation.
        """
        required_results = {"energy", "forces"}
        for trajectory_path in self.training_trajectory_paths():
            for frame_index, atoms in enumerate(read_atoms_trajectory(trajectory_path)):
                available_results = set(getattr(atoms.calc, "results", {})) if atoms.calc else set()
                if not required_results.issubset(available_results):
                    raise ValueError(
                        f"Training frame {frame_index} in {trajectory_path} is missing an energy or forces; "
                        "every configuration in the training database must be labelled."
                    )

    # ----------------------------------------------------------------- restart
    def resume_point(self, restart_from_stage: str = "auto") -> Tuple[int, Stage]:
        """Determine the (epoch, stage) at which the loop should (re)enter.

        This is a pure query. For a forced (non-auto) restart the caller should additionally call
        ``reset_epoch_to_stage`` so the artifacts of the discarded stages are cleared before rerunning.

        Args:
            restart_from_stage: 'auto' (infer from what is on disk) or an explicit 'driver'/'generate'/
                'oracle'/'train' override that forces the stage on the latest epoch (validating its
                prerequisites exist).

        Returns:
            (epoch, stage): the epoch number and the stage to start executing.
        """
        if restart_from_stage == "auto":
            return self._auto_resume_stage()
        return self._forced_resume_stage(restart_from_stage)

    def _driver_resume_stage(self) -> Tuple[int, Stage]:
        """Return the (epoch, stage) at which a driver run should (re)enter.

        Epoch 0 (precomputation) when nothing is trained yet, the next round when the latest one is complete,
        or a re-run of the latest round otherwise.
        """
        latest = self.epoch
        if latest == 0:
            # No rounds yet: start round 1 if precomputation (epoch 0) is done, else precompute.
            return (1, Stage.DRIVER) if self.is_model_committed(0) else (0, Stage.DRIVER)
        if self.is_model_committed(latest):
            return latest + 1, Stage.DRIVER
        return latest, Stage.DRIVER

    def _auto_resume_stage(self) -> Tuple[int, Stage]:
        """Return (epoch, stage) inferred from the first incomplete stage of the latest epoch."""
        latest = self.epoch
        if latest == 0 or self.is_model_committed(latest):
            return self._driver_resume_stage()
        if self.is_oracle_committed(latest):
            return latest, Stage.TRAIN
        if self.is_generate_committed(latest):
            return latest, Stage.ORACLE
        if self.is_dynamic_committed(latest):
            return latest, Stage.GENERATE
        return self._driver_resume_stage()

    def _forced_resume_stage(self, restart_from_stage: str) -> Tuple[int, Stage]:
        """Return (epoch, stage) for an explicit override, validating the stage's prerequisites exist."""
        if restart_from_stage == "driver":
            return self._driver_resume_stage()

        latest = self.epoch
        if latest == 0:
            raise ValueError(
                f"Cannot restart from '{restart_from_stage}': the database has no epoch to resume."
            )
        if restart_from_stage == "generate":
            if not self.is_dynamic_committed(latest):
                raise ValueError(
                    f"Cannot restart from 'generate': epoch {latest} has no committed 'dynamic.traj'."
                )
            return latest, Stage.GENERATE
        if restart_from_stage == "oracle":
            if not self.is_generate_committed(latest):
                raise ValueError(
                    f"Cannot restart from 'oracle': epoch {latest} has no committed 'generate.traj'."
                )
            return latest, Stage.ORACLE
        if restart_from_stage == "train":
            if not self.is_oracle_committed(latest):
                raise ValueError(
                    f"Cannot restart from 'train': epoch {latest} has no committed 'oracle.traj'."
                )
            return latest, Stage.TRAIN
        raise ValueError(
            f"Unknown restart_from_stage '{restart_from_stage}'; expected 'auto', 'driver', 'generate', "
            "'oracle' or 'train'."
        )

    def reset_epoch_to_stage(self, epoch: int, stage: Stage) -> None:
        """Delete the artifacts at or after ``stage`` for ``epoch`` so a forced restart reruns them cleanly."""
        artifacts_from_stage = {
            Stage.DRIVER: ["dynamic", "dynamic.traj", "generate.traj", "oracle", "oracle.traj", "model"],
            Stage.GENERATE: ["generate.traj", "oracle", "oracle.traj", "model"],
            Stage.ORACLE: ["oracle", "oracle.traj", "model"],
            Stage.TRAIN: ["model"],
        }
        epoch_directory = self._root / f"epoch_{epoch}"
        for name in artifacts_from_stage[stage]:
            target = epoch_directory / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.is_file():
                target.unlink()

    def rollback_last_epoch(self) -> int:
        """Delete the highest epoch folder (e.g. after an MLIP collapse); return the removed epoch (0 if none)."""
        latest = self.epoch
        if latest == 0:
            return 0
        shutil.rmtree(self._root / f"epoch_{latest}")
        return latest

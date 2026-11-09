"""Base class for a trainable interatomic potential used in the active learning loop."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from ase import Atoms
from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential
from diffusion_for_multi_scale_molecular_dynamics.io.training_database import \
    TrainingDatabase
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip_trainer import \
    BaseMLIPTrainer
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import (
    BaseSinglePointCalculator, SinglePointCalculation)
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_single_point_calculator import \
    LammpsSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_conversion import \
    to_ase_atoms
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import (
    atoms_per_element, create_perturbed_structures, label_configurations)


class BaseMLIP(ABC):
    """A trainable interatomic potential for the active learning loop.

    A MLIP is built from a trainer (the trainable core) and it caches the LAMMPS potential
    that the trainer exports whenever the model is deployed.
    """

    name = "MLIP"  # human-readable backend name, used in logs; overridden by each concrete MLIP.
    training_program_name = "the model"  # what the training step launches (log); overridden per backend.

    def __init__(
        self,
        trainer: BaseMLIPTrainer,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
    ):
        """Init method.

        Args:
            trainer: the trainable core that learns and exports a LAMMPS potential.
            lammps_runner: runner used to evaluate the deployed potential (e.g. for training metrics).
        """
        self._trainer = trainer
        self._lammps_runner = lammps_runner
        self._lammps_potential: Optional[LammpsPotential] = None
        self._model_file: Optional[Path] = None
        # Training-set metrics of the deployed model, computed once and reused (state file + logs); a new
        # deploy invalidates it.
        self._cached_training_metrics: Optional[Dict] = None
        # Specific descriptor information; populated by each MLIP subclass.
        self.descriptors: Dict[str, Any] = {}

    @property
    def lammps_potential(self) -> LammpsPotential:
        """The currently deployed LAMMPS potential."""
        if self._lammps_potential is None:
            raise RuntimeError("The MLIP has not been deployed yet; call prepare_mlip_first_round or train first.")
        return self._lammps_potential

    @property
    def model_file(self) -> Optional[Path]:
        """Path to the current reloadable model checkpoint (None until the model has been trained)."""
        return self._model_file

    @property
    def training_database(self) -> Optional[TrainingDatabase]:
        """The training database backing this MLIP's trainer (None until one is attached)."""
        return self._trainer.training_database

    def attach_training_database(self, training_database: TrainingDatabase) -> None:
        """Attach the training database; the active learning loop calls this at the start of a run."""
        self._trainer.set_training_database(training_database)

    def add_labelled_structure(
        self, single_point_calculation: SinglePointCalculation, active_environment_indices: List[int]
    ) -> None:
        """Add a labelled structure to the training set."""
        self._trainer.add_labelled_structure(single_point_calculation, active_environment_indices)

    def minimum_number_of_training_environments(self) -> Dict[str, int]:
        """Per-element atomic-environment floor the active set needs (element symbol -> environment count).

        The default is an empty mapping (no D-optimality active set, e.g. FLARE); a D-optimality MLIP (MTP,
        GRACE-FS) redefines this. GRACE-FS is genuinely per-element; MTP's single pooled floor is spread over
        its species.
        """
        return {}

    def _greedy_augmentation_seeds(
        self, provided_configurations: List[Atoms], minimum_environments_per_element: Dict[str, int]
    ) -> List[Atoms]:
        """Greedily choose which provided configurations to rattle to reach the per-element active-set floor.

        Selection metric - the normalized dot product (cosine similarity) between the number missing per
        element and the number provided by each configuration, iteratively repeated until every element
        deficiency is resolved.

        Returns:
            the chosen seed configurations, one entry per rattled copy to create (a configuration may repeat).
        """
        def cosine_similarity(vector: np.ndarray, other: np.ndarray) -> float:
            norm_product = float(np.linalg.norm(vector) * np.linalg.norm(other))
            return 0.0 if norm_product == 0.0 else float(np.dot(vector, other) / norm_product)

        element_order = sorted(minimum_environments_per_element)
        minimum_per_element = np.array(
            [minimum_environments_per_element[element] for element in element_order], dtype=float
        )
        atoms_per_configuration = [
            atoms_per_element(configuration, element_order) for configuration in provided_configurations
        ]

        provided_environments = (
            np.sum(atoms_per_configuration, axis=0) if atoms_per_configuration else np.zeros(len(element_order))
        )
        missing = np.maximum(minimum_per_element - provided_environments, 0)

        # Point 2: every still-needed element must be reachable by rattling some provided configuration.
        for index, element in enumerate(element_order):
            if missing[index] > 0 and all(counts[index] == 0 for counts in atoms_per_configuration):
                raise ValueError(
                    f"No provided configuration contains '{element}', which the active set still needs "
                    f"({int(missing[index])} more environments). Provide a configuration containing it."
                )

        selected_seeds: List[Atoms] = []
        while np.any(missing > 0):
            best_index = max(
                range(len(provided_configurations)),
                key=lambda index: (
                    cosine_similarity(atoms_per_configuration[index], missing),
                    -len(provided_configurations[index]),  # tie-break: fewer atoms is cheaper to label
                ),
            )
            selected_seeds.append(provided_configurations[best_index])
            missing = np.maximum(missing - atoms_per_configuration[best_index], 0)
        return selected_seeds

    def prepare_training_set(
        self,
        provided_configurations: List[Atoms],
        single_point_calculator: BaseSinglePointCalculator,
        standard_deviation: float = 0.05,
    ) -> int:
        """Seed this MLIP's training database from the provided configuration(s) for precomputation.

        Assembles the training configurations (augmenting the provided configuration(s) with oracle-labelled
        perturbations when they are too few) and writes the provided and training configurations to the
        database. Does nothing when the training set already covers the minimum number of training
        environments (e.g. on a restart).

        Args:
            provided_configurations: the (labelled) seed configurations.
            single_point_calculator: the oracle used to label the perturbed copies.
            standard_deviation: standard deviation (Angstrom) of the Gaussian displacements when augmenting.

        Returns:
            the number of augmented (perturbed, oracle-labelled) configurations added beyond the provided ones.
        """
        database = self.training_database
        minimum_environments_per_element = self.minimum_number_of_training_environments()
        element_order = sorted(minimum_environments_per_element)
        labelled_environments = sum(
            (atoms_per_element(atoms, element_order) for atoms in database.labelled_atoms),
            np.zeros(len(element_order)),
        )
        minimum_per_element = np.array(
            [minimum_environments_per_element[element] for element in element_order], dtype=float
        )
        if np.all(labelled_environments >= minimum_per_element):
            return 0

        training_configurations = self.create_training_configurations(
            provided_configurations, single_point_calculator, standard_deviation
        )
        database.write_provided_configurations(provided_configurations)
        database.append_training_configurations(training_configurations)
        return len(training_configurations) - len(provided_configurations)

    def create_training_configurations(
        self,
        provided_configurations: List[Atoms],
        single_point_calculator: BaseSinglePointCalculator,
        standard_deviation: float = 0.05,
    ) -> List[Atoms]:
        """Assemble the labelled training set from the provided configuration(s).

        The provided configuration(s) must already be labelled (energy and forces). When they hold fewer
        environments than the D-optimality minimum, they are augmented with perturbed, oracle-labelled copies.

        Args:
            provided_configurations: the (labelled) seed configurations.
            single_point_calculator: the oracle used to label the perturbed copies.
            standard_deviation: standard deviation (Angstrom) of the Gaussian displacements when augmenting.

        Returns:
            the labelled training configurations (provided plus any augmented, oracle-labelled copies).
        """
        required_results = {"energy", "forces"}
        for configuration in provided_configurations:
            available_results = set(getattr(configuration.calc, "results", {})) if configuration.calc else set()
            if not required_results.issubset(available_results):
                raise ValueError(
                    "The provided configurations must be labelled (carry an energy and forces) to seed the "
                    "training set. Label them first with "
                    "utils.structure_utils.label_configurations(configurations, oracle)."
                )

        training_configurations = list(provided_configurations)
        seeds_to_rattle = self._greedy_augmentation_seeds(
            provided_configurations, self.minimum_number_of_training_environments()
        )
        if seeds_to_rattle:
            perturbed_configurations = [
                create_perturbed_structures(seed, standard_deviation, 1)[0] for seed in seeds_to_rattle
            ]
            training_configurations += label_configurations(perturbed_configurations, single_point_calculator)
        return training_configurations

    def prepare_mlip_first_round(self, output_directory: Path) -> None:
        """Deploy the pretrained model so it can be run before any training happens this campaign."""
        self._deploy(output_directory)

    def _deploy(self, output_directory: Path) -> None:
        """Write the model into output_directory and cache the resulting LAMMPS potential."""
        self._lammps_potential = self._trainer.write_checkpoint(output_directory)
        self._cached_training_metrics = None  # a new model makes any cached metrics stale

    def load(self, model_directory: Path) -> None:
        """Load an already-trained model from model_directory into a runnable potential (no fitting).

        Used to resume a campaign straight off disk. Retraining later refits from the training database, so
        this only needs to restore the deployed potential, not the trainer's state.
        """
        raise NotImplementedError("must be implemented in a child class.")

    def calculate(
        self, configurations: Union[Structure, Atoms, List[Union[Structure, Atoms]]]
    ) -> List[SinglePointCalculation]:
        """Evaluate configurations with the deployed potential.

        Args:
            configurations: a single configuration (pymatgen Structure or ase.Atoms), or a list of them.

        Returns:
            a list of SinglePointCalculation.
        """
        if not isinstance(configurations, list):
            configurations = [configurations]

        calculator = LammpsSinglePointCalculator(
            lammps_potential=self.lammps_potential, lammps_runner=self._lammps_runner
        )
        return [calculator.calculate(to_ase_atoms(configuration)) for configuration in configurations]

    def training_set_state(self) -> Dict:
        """Training-set provenance for the state file: the source trajectories, the epoch and the counts.

        Returns an empty dict when no training database is attached.
        """
        database = self.training_database
        if database is None:
            return {}
        labelled_atoms = database.labelled_atoms
        metrics = self.training_metrics()
        return dict(
            epoch=database.epoch,
            number_of_training_configurations=len(labelled_atoms),
            number_of_training_atomic_environments=sum(len(atoms) for atoms in labelled_atoms),
            rmse_energy_meV_per_atom=None if metrics["rmse_energy"] is None else 1000 * metrics["rmse_energy"],
            rmse_forces_meV_per_angstrom=None if metrics["rmse_forces"] is None else 1000 * metrics["rmse_forces"],
            training_files=[str(path) for path in database.training_trajectory_paths()],
        )

    def training_metrics(self, reference_atoms: Optional[List[Atoms]] = None) -> Dict:
        """Return accuracy metrics (configuration count, per-atom energy RMSE, forces RMSE) of the model.

        The metrics are computed over the training set by default (and cached until the next deploy); pass
        reference_atoms to evaluate the potential against a given set of labelled ase.Atoms instead.
        """
        if reference_atoms is None and self._cached_training_metrics is not None:
            return self._cached_training_metrics

        if reference_atoms is None:
            calculations = self._trainer.labelled_calculations
        else:
            calculations = [
                SinglePointCalculation(calculation_type="reference",
                                       atoms=atoms,
                                       forces=atoms.get_forces(),
                                       energy=atoms.get_potential_energy())
                for atoms in reference_atoms
            ]

        if not calculations:  # e.g. round 0 of active learning
            metrics = dict(n_training_conf=0, n_training_atomic_environments=0,
                           rmse_energy=None, rmse_forces=None)
        else:
            predictions = self.calculate([calculation.atoms for calculation in calculations])

            energy_errors = []
            force_errors = []
            for prediction, calculation in zip(predictions, calculations):
                # Per-atom energy error (atom count = number of force vectors), so the energy RMSE is per atom.
                number_of_atoms = len(calculation.forces)
                energy_errors.append((prediction.energy - calculation.energy) / number_of_atoms)
                force_errors.append((np.asarray(prediction.forces) - np.asarray(calculation.forces)).ravel())

            metrics = dict(
                n_training_conf=len(calculations),
                n_training_atomic_environments=sum(len(calc.forces) for calc in calculations),
                rmse_energy=float(np.sqrt(np.mean(np.square(energy_errors)))),
                rmse_forces=float(np.sqrt(np.mean(np.square(np.concatenate(force_errors))))),
            )

        if reference_atoms is None:
            self._cached_training_metrics = metrics
        return metrics

    @abstractmethod
    def train(self, output_directory: Path) -> None:
        """Train the model, deploy it and write a checkpoint into output_directory."""
        raise NotImplementedError("must be implemented in a child class.")

    @abstractmethod
    def write_state_yaml(self, output_path: Path) -> None:
        """Write a yaml with the current model_file, unc_file, lammps_potential_file and hyperparameters."""
        raise NotImplementedError("must be implemented in a child class.")

    @classmethod
    @abstractmethod
    def load_checkpoint(cls, checkpoint_path: Path) -> "BaseMLIP":
        """Reconstruct the MLIP from a checkpoint."""
        raise NotImplementedError("must be implemented in a child class.")

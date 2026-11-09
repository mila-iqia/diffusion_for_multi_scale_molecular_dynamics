"""GRACE-FS trainer."""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.grace import \
    GracePotential
from diffusion_for_multi_scale_molecular_dynamics.io.mlip import \
    write_grace_pkl
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip_trainer import \
    BaseMLIPTrainer
from diffusion_for_multi_scale_molecular_dynamics.mlip.grace.grace_configuration import \
    GraceConfiguration
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    SinglePointCalculation

# gracemaker exports under FS_model.* ('-sf'); older builds also drop saved_model.* ('-s').
_EXPORT_STEMS = ("FS_model", "saved_model")


class GraceTrainer(BaseMLIPTrainer):
    """Fit a GRACE-FS model with gracemaker and deploy it (model .yaml + active set .asi) as a LAMMPS potential."""

    def __init__(
        self,
        grace_configuration: GraceConfiguration,
        initial_configuration: Optional[SinglePointCalculation] = None,
        gracemaker_executable_path: Optional[Path] = None,
        pace_activeset_executable_path: Optional[Path] = None,
        training_database=None,
    ):
        """Init method.

        Args:
            grace_configuration: the GRACE-FS model definition (mapped onto the gracemaker input.yaml).
            initial_configuration: the labelled configuration used as the (separate) gracemaker test set. When
                None, the first training configuration is reused as the test set.
            gracemaker_executable_path: path to 'gracemaker'; looked up on PATH if None.
            pace_activeset_executable_path: path to 'pace_activeset'; looked up on PATH if None.
            training_database: the training set (single source of truth); may be attached later.
        """
        super().__init__(training_database)
        self._gracemaker_executable_path = self._resolve_executable(gracemaker_executable_path, "gracemaker")
        self._pace_activeset_executable_path = self._resolve_executable(
            pace_activeset_executable_path, "pace_activeset"
        )
        self._configuration = grace_configuration
        self._initial_configuration = initial_configuration
        # Persistent fit directory: kept across fit() calls so gracemaker can restart from its checkpoint.
        self._fit_directory = Path(tempfile.mkdtemp(prefix="grace_fit_"))
        self._has_fitted = False

    @property
    def configuration(self) -> GraceConfiguration:
        """The GRACE-FS configuration."""
        return self._configuration

    @property
    def _seed_directory(self) -> Path:
        """The gracemaker output directory for the configured seed."""
        return self._fit_directory / "seed" / str(self._configuration.seed)

    def _add_labelled_structure_to_model(
        self, single_point_calculation: SinglePointCalculation, active_environment_indices: List[int]
    ) -> None:
        """Nothing is folded in per structure: GRACE is fit in batch, over all labelled structures, at fit time."""
        if single_point_calculation.uncertainties is not None:
            raise ValueError("Only ground-truth single-point calculations (without uncertainties) can be added.")

    def create_test_set(self, test_set_path: Path) -> None:
        """Write the gracemaker test set: the provided initial configuration, or the first training config."""
        if self._initial_configuration is not None:
            test_atoms = self._initial_configuration.to_atoms()
        else:
            # No explicit test configuration: reuse the first training config (gracemaker needs a test set).
            assert self._training_database and self._training_database.labelled_atoms, (
                "GraceTrainer needs either an initial_configuration or a non-empty training database to build "
                "the gracemaker test set."
            )
            test_atoms = self._training_database.labelled_atoms[0]
        write_grace_pkl([test_atoms], test_set_path)

    def fit(self) -> None:
        """Fit the GRACE-FS model with gracemaker and export it (the active set is built at deploy time)."""
        training_atoms = self._training_database.labelled_atoms if self._training_database else []
        if not training_atoms:
            raise RuntimeError("Cannot fit a GRACE-FS model with no labelled structures.")

        training_set_path = self._fit_directory / "train.pkl.gz"
        test_set_path = self._fit_directory / "test.pkl.gz"
        write_grace_pkl(training_atoms, training_set_path)
        if not test_set_path.exists():
            self.create_test_set(test_set_path)

        input_yaml_path = self._fit_directory / "input.yaml"
        self._write_input_yaml(input_yaml_path, training_set_path, test_set_path)

        # Fit: cold on the first call, then resume from the previous checkpoint ('-rl') on later calls.
        train_command = [str(self._gracemaker_executable_path)]
        if self._has_fitted:
            train_command.append("-rl")
        train_command.append(input_yaml_path.name)
        self._run(train_command, self._fit_directory)
        self._has_fitted = True

        # Export the FS model (yaml) from the latest checkpoint (a separate gracemaker invocation).
        self._run(
            [str(self._gracemaker_executable_path), "-rl", "-s", "-sf", input_yaml_path.name], self._fit_directory
        )

    @property
    def exported_model_path(self) -> Path:
        """Path to the exported GRACE-FS model (.yaml) from the latest fit."""
        return self._resolve_export(".yaml")

    def write_checkpoint(self, output_directory: Path) -> GracePotential:
        """Build the active set, then write the model (.yaml), active set (.asi) and a copy of the seed folder."""
        if not self._has_fitted:
            raise RuntimeError("The GRACE-FS model has not been fitted yet; call fit first.")

        # The active set is a deployment artifact (needed for the extrapolation grade) and requires enough
        # atomic environments, so it is built here rather than during fit().
        training_set_path = self._fit_directory / "train.pkl.gz"
        self._run(
            [str(self._pace_activeset_executable_path), "-d", str(training_set_path.resolve()),
             str(self.exported_model_path.resolve())],
            self._seed_directory,
        )

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        model_file_path = output_directory / "model.yaml"
        active_set_file_path = output_directory / "model.asi"
        shutil.copy(self._resolve_export(".yaml"), model_file_path)
        shutil.copy(self._resolve_export(".asi"), active_set_file_path)
        # Bookkeeping + restart: copy the whole 'seed' tree (i.e. seed/<seed>/checkpoints/...) so that a trainer
        # rebuilt via load_checkpoint can resume with gracemaker -rl.
        shutil.copytree(self._fit_directory / "seed", output_directory / "seed", dirs_exist_ok=True)

        return GracePotential(model_file_path=model_file_path, active_set_file_path=active_set_file_path)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Path,
        grace_configuration: GraceConfiguration,
        initial_configuration: Optional[SinglePointCalculation] = None,
        gracemaker_executable_path: Optional[Path] = None,
        pace_activeset_executable_path: Optional[Path] = None,
        training_database=None,
    ) -> "GraceTrainer":
        """Rebuild a trainer from a checkpoint, restoring the seed folder so gracemaker can resume from it."""
        trainer = cls(
            grace_configuration=grace_configuration,
            initial_configuration=initial_configuration,
            gracemaker_executable_path=gracemaker_executable_path,
            pace_activeset_executable_path=pace_activeset_executable_path,
            training_database=training_database,
        )
        trainer.restore_from_checkpoint(checkpoint_path)
        return trainer

    def restore_from_checkpoint(self, checkpoint_directory: Path) -> None:
        """Restore the seed folder from a committed model directory so the next fit resumes with '-rl'.

        The '-rl' warm start drastically shortens gracemaker's fitting, so it must survive a restart: the
        committed model directory carries the seed tree written by ``write_checkpoint``.
        """
        checkpoint_seed_directory = Path(checkpoint_directory) / "seed"
        if checkpoint_seed_directory.is_dir():
            shutil.copytree(checkpoint_seed_directory, self._fit_directory / "seed", dirs_exist_ok=True)
            self._has_fitted = True

    @staticmethod
    def _resolve_executable(executable_path: Optional[Path], name: str) -> Path:
        """Resolve an executable from an explicit path or PATH, raising if it cannot be found."""
        if executable_path is None:
            executable_path = shutil.which(name)
        if executable_path is None:
            raise ValueError(f"No {name} executable was provided and none was found on PATH.")
        executable_path = Path(executable_path)
        if not executable_path.is_file():
            raise ValueError(f"The {name} executable '{executable_path}' does not exist. Review input.")
        return executable_path

    def _resolve_export(self, suffix: str) -> Path:
        """Return the existing seed-folder export with the given suffix ('.yaml' or '.asi'), preferring FS_model."""
        for stem in _EXPORT_STEMS:
            candidate = self._seed_directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        expected = " or ".join(f"{stem}{suffix}" for stem in _EXPORT_STEMS)
        raise FileNotFoundError(f"gracemaker export not found in {self._seed_directory} (expected {expected}).")

    def _write_input_yaml(self, path: Path, training_set_path: Path, test_set_path: Path) -> None:
        """Emit the gracemaker input.yaml from the configuration, pointing at the training and test datasets."""
        import yaml

        configuration = self._configuration
        input_dictionary = dict(
            seed=configuration.seed,
            cutoff=configuration.cutoff,
            data=dict(
                filename=training_set_path.name,
                test_filename=test_set_path.name,
                reference_energy=configuration.reference_energy,
            ),
            potential=dict(
                elements=list(configuration.elements),
                preset=configuration.preset,
                kwargs=configuration.model_kwargs,
                scale=configuration.scale,
            ),
            fit=dict(
                loss=configuration.loss,
                optimizer=configuration.optimizer,
                opt_params=configuration.opt_params,
                target_total_updates=configuration.target_total_updates,
                batch_size=configuration.batch_size,
                test_batch_size=configuration.test_batch_size,
                jit_compile=configuration.jit_compile,
            ),
        )
        with open(path, "w") as file_descriptor:
            yaml.dump(input_dictionary, file_descriptor, sort_keys=False)

    @staticmethod
    def _run(command: List[str], working_directory: Path) -> None:
        """Run a subprocess in working_directory, surfacing its output on failure."""
        result = subprocess.run(
            command, cwd=working_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{command[0]} failed (exit code {result.returncode}). Command: {' '.join(command)}\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )

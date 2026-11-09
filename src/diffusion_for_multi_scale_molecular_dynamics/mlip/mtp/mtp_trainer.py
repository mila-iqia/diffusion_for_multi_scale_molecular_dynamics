"""Moment Tensor Potential trainer."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.mtp import \
    MtpPotential
from diffusion_for_multi_scale_molecular_dynamics.io.mlip import write_mtp_cfg
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip_trainer import \
    BaseMLIPTrainer
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_configuration import \
    MtpConfiguration
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    SinglePointCalculation
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_conversion import \
    to_pymatgen_structure


class MtpTrainer(BaseMLIPTrainer):
    """Fit a Moment Tensor Potential with the MLIP-3 'mlp train' command and deploy it as a LAMMPS potential."""

    def __init__(self, mtp_configuration: MtpConfiguration, mlp_executable_path: Optional[Path] = None,
                 training_database=None):
        """Init method.

        Args:
            mtp_configuration: the MTP model definition (level, cutoff, weights, training parameters).
            mlp_executable_path: path to the MLIP-3 'mlp' executable; if None, it is looked up on PATH.
            training_database: the training set (single source of truth); may be attached later.
        """
        super().__init__(training_database)
        if mlp_executable_path is None:
            mlp_executable_path = shutil.which("mlp")
        if mlp_executable_path is None:
            raise ValueError("No mlp executable was provided and none was found on PATH ('which mlp' failed).")
        mlp_executable_path = Path(mlp_executable_path)
        if not mlp_executable_path.is_file():
            raise ValueError(f"The mlp executable '{mlp_executable_path}' does not exist. Review input.")

        self._configuration = mtp_configuration
        self._mlp_executable_path = mlp_executable_path
        self._fitted_potential: Optional[bytes] = None
        self._training_log: Optional[str] = None  # captured 'mlp train' output of the last fit

    @property
    def configuration(self) -> MtpConfiguration:
        """The MTP configuration, with the level-determined parameters filled in once fitted."""
        return self._configuration

    @property
    def template_path(self) -> Path:
        """Path to the level template shipped with the package (its header carries the basis descriptors)."""
        return self._template_path()

    def _add_labelled_structure_to_model(
        self,
        single_point_calculation: SinglePointCalculation,
        active_environment_indices: List[int],
    ) -> None:
        """Nothing is folded in per structure: MTP is fit in batch, over all labelled structures, at fit time."""
        if single_point_calculation.uncertainties is not None:
            raise ValueError(
                "Only ground-truth single-point calculations (without uncertainties) can be added."
            )

    def fit(self) -> None:
        """Fit the MTP with MLIP-3 and read the level-determined parameters back into the configuration."""
        from maml.utils import check_structures_forces_stresses, pool_from

        labelled_calculations = self.labelled_calculations
        if not labelled_calculations:
            raise RuntimeError("Cannot fit an MTP with no labelled structures.")

        structures = [to_pymatgen_structure(calculation.atoms) for calculation in labelled_calculations]
        forces = [np.asarray(calculation.forces) for calculation in labelled_calculations]
        energies = [calculation.energy for calculation in labelled_calculations]
        checked_structures, checked_forces, _ = check_structures_forces_stresses(
            structures, forces, None
        )
        training_pool = pool_from(checked_structures, energies, checked_forces)

        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as work_directory:
            os.chdir(work_directory)
            try:
                write_mtp_cfg(training_pool, self._configuration.elements, Path("train.cfg"))

                starting_potential_path = Path(f"{self._configuration.level:02d}.almtp")
                if self._fitted_potential is not None:
                    # Warm start: resume from the previous fit's coefficients (mlp --init-params='same').
                    starting_potential_path.write_bytes(self._fitted_potential)
                else:
                    # First fit: cold start from the level template, then set its max_dist.
                    shutil.copy(self._template_path(), starting_potential_path)
                    self._configuration.write_to_file(starting_potential_path)

                fitted_potential_path = Path("pot.almtp")
                self._run_mlp_train(starting_potential_path, Path("train.cfg"), fitted_potential_path)

                self._fitted_potential = fitted_potential_path.read_bytes()
                self._configuration.read_from_file(fitted_potential_path)
            finally:
                os.chdir(original_directory)

    def restore_from_checkpoint(self, checkpoint_directory: Path) -> None:
        """Restore the fitted potential from a committed model directory so the next fit warm-starts from it."""
        self._fitted_potential = (Path(checkpoint_directory) / "potential.almtp").read_bytes()

    def write_checkpoint(self, output_directory: Path) -> MtpPotential:
        """Write the fitted potential into output_directory and return the deployed LAMMPS potential."""
        if self._fitted_potential is None:
            raise RuntimeError("The MTP has not been fitted yet; call fit first.")

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        mtp_file_path = output_directory / "potential.almtp"
        mtp_file_path.write_bytes(self._fitted_potential)
        return MtpPotential(mtp_file_path=mtp_file_path)

    def write_training_log(self, output_directory: Path) -> None:
        """Write the captured 'mlp train' output (its progress and per-step metrics) to training.log."""
        if self._training_log is None:
            return
        (Path(output_directory) / "training.log").write_text(self._training_log)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Path,
        mtp_configuration: MtpConfiguration,
        mlp_executable_path: Optional[Path] = None,
        training_database=None,
    ) -> "MtpTrainer":
        """Rebuild a trainer from a fitted MTP file.

        The MTP file does not record the level or the element symbols, so the configuration must be provided.
        """
        trainer = cls(
            mtp_configuration=mtp_configuration, mlp_executable_path=mlp_executable_path,
            training_database=training_database,
        )
        trainer._fitted_potential = Path(checkpoint_path).read_bytes()
        trainer._configuration.read_from_file(Path(checkpoint_path))
        return trainer

    def _template_path(self) -> Path:
        """Path to the level template shipped with the package."""
        return (
            Path(__file__).parent.parent
            / "MTP_templates"
            / f"{self._configuration.level:02d}.almtp"
        )

    def _run_mlp_train(
        self,
        template_path: Path,
        training_configuration_path: Path,
        fitted_potential_path: Path,
    ) -> None:
        """Run 'mlp train' in the current working directory, surfacing its output on failure."""
        training_params = self._configuration.training_params
        commands = [
            str(self._mlp_executable_path),
            "train",
            str(template_path),
            str(training_configuration_path),
            f"--save_to={fitted_potential_path}",
            f"--iteration_limit={training_params['max_iter']}",
            "--al_mode=nbh",
            f"--energy-weight={self._configuration.energy_weight}",
            f"--force-weight={self._configuration.force_weight}",
            f"--stress-weight={self._configuration.stress_weight}",
            f"--site-en-weight={self._configuration.site_en_weight}",
            f"--init-params={training_params['init_params']}",
            f"--scale-by-force={training_params['scale_by_force']}",
            f"--bfgs-conv-tol={training_params['bfgs_conv_tol']}",
        ]
        result = subprocess.run(
            commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        self._training_log = result.stdout
        if result.stderr:
            self._training_log += f"\n--- stderr ---\n{result.stderr}"
        if result.returncode != 0:
            raise RuntimeError(
                f"mlp train failed (exit code {result.returncode}). Command: {' '.join(commands)}\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )

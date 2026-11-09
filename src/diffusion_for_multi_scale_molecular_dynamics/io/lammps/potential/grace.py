"""Graph Atomic Cluster Expansion (GRACE-FS) potential run through the LAMMPS grace/fs pair_style."""

from pathlib import Path
from typing import List, Optional

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential


class GracePotential(LammpsPotential):
    """GRACE-FS potential (trained with gracemaker) run through the LAMMPS grace/fs pair_style."""

    calculation_type = "grace"
    name = "GRACE"

    def __init__(self, model_file_path: Path, active_set_file_path: Optional[Path] = None):
        """Init method.

        Args:
            model_file_path: path to the exported GRACE-FS model (.yaml).
            active_set_file_path: path to the active set (.asi); required only to compute the extrapolation
                grade (i.e. when running with uncertainty).
        """
        if not model_file_path.is_file():
            raise ValueError(f"The file '{model_file_path}' does not exist. Review input.")
        if active_set_file_path is not None and not active_set_file_path.is_file():
            raise ValueError(f"The file '{active_set_file_path}' does not exist. Review input.")

        self._model_file_path = model_file_path.resolve()
        self._active_set_file_path = None if active_set_file_path is None else active_set_file_path.resolve()

    @property
    def model_file_path(self) -> Path:
        """Path to the exported GRACE-FS model."""
        return self._model_file_path

    @property
    def active_set_file_path(self) -> Optional[Path]:
        """Path to the active set used for the extrapolation grade (None if not provided)."""
        return self._active_set_file_path

    def interaction_commands(self, elements_string: str, with_uncertainty: bool = False) -> List[str]:
        """Return the GRACE-FS interaction commands.

        Without uncertainty it uses the plain 'grace/fs' pair_style; with uncertainty it uses
        'grace/fs extrapolation', passes the active set and adds the extrapolation-grade fix.
        """
        if with_uncertainty and self._active_set_file_path is None:
            raise ValueError("An active set (.asi) is required to run GRACE-FS with uncertainty.")

        pair_style = "grace/fs extrapolation" if with_uncertainty else "grace/fs"
        pair_coeff_files = str(self._model_file_path)
        if with_uncertainty:
            pair_coeff_files += f" {self._active_set_file_path}"

        commands = [
            f"pair_style {pair_style}",
            f"pair_coeff * * {pair_coeff_files} {elements_string}",
        ]
        if with_uncertainty:
            commands.append("fix grace_gamma all pair 1 grace/fs gamma 1")
        return commands

    def dump_fields(self, with_uncertainty: bool = False) -> List[str]:
        """Return the per-atom fields written to the main dump."""
        fields = super().dump_fields(with_uncertainty=with_uncertainty)
        if with_uncertainty:
            fields = fields + ["f_grace_gamma"]
        return fields

    def uncertainty_field(self) -> str:
        """Return the per-atom uncertainty (extrapolation grade) column name."""
        return "f_grace_gamma"

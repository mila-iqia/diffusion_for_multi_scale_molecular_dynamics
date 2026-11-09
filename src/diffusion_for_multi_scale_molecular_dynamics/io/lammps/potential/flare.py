"""FLARE mapped potential."""

from pathlib import Path
from typing import List

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential


class FlarePotential(LammpsPotential):
    """FLARE mapped potential."""

    calculation_type = "flare"
    name = "FLARE"

    def __init__(self, pair_coeff_file_path: Path, mapped_uncertainty_file_path: Path):
        """Init method.

        Args:
            pair_coeff_file_path: path to the mapped FLARE coefficients.
            mapped_uncertainty_file_path: path to the mapped FLARE uncertainty coefficients.
        """
        assert pair_coeff_file_path.is_file(), \
            f"The file '{pair_coeff_file_path}' does not exist. Review input."
        assert mapped_uncertainty_file_path.is_file(), \
            f"The file '{mapped_uncertainty_file_path}' does not exist. Review input."
        self._pair_coeff_path = pair_coeff_file_path.resolve()
        self._mapped_uncertainty_path = mapped_uncertainty_file_path.resolve()

    @property
    def pair_coeff_file_path(self) -> Path:
        """Path to the mapped FLARE coefficients."""
        return self._pair_coeff_path

    @property
    def mapped_uncertainty_file_path(self) -> Path:
        """Path to the mapped FLARE uncertainty coefficients."""
        return self._mapped_uncertainty_path

    def interaction_commands(self, elements_string: str, with_uncertainty: bool = False) -> List[str]:
        """Return the FLARE interaction commands."""
        commands = [
            "pair_style flare",
            f"pair_coeff * * {self._pair_coeff_path}",
        ]
        if with_uncertainty:
            commands.append(f"compute unc_at all flare/std/atom {self._mapped_uncertainty_path}")
        return commands

    def dump_fields(self, with_uncertainty: bool = False) -> List[str]:
        """Return the per-atom fields written to the main dump."""
        fields = super().dump_fields(with_uncertainty=with_uncertainty)
        if with_uncertainty:
            fields = fields + ["c_unc_at"]
        return fields

    def uncertainty_field(self) -> str:
        """Return the per-atom uncertainty column name."""
        return "c_unc_at"

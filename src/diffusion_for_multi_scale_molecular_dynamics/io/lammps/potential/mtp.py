"""Moment Tensor Potential run through the lammps-mtp-kokkos interface."""

from pathlib import Path
from typing import List

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential


class MtpPotential(LammpsPotential):
    """Moment Tensor Potential run through the lammps-mtp-kokkos pair_style."""

    calculation_type = "mtp"
    name = "MTP"

    def __init__(self, mtp_file_path: Path, kokkos: bool = False):
        """Init method.

        Args:
            mtp_file_path: path to the trained MTP potential file.
            kokkos: whether to use the Kokkos-accelerated pair_style (not implemented yet).
        """
        if not mtp_file_path.is_file():
            raise ValueError(f"The file '{mtp_file_path}' does not exist. Review input.")
        if kokkos:
            raise NotImplementedError("The Kokkos MTP pair_style is not implemented yet.")

        self._mtp_file_path = mtp_file_path.resolve()
        self._kokkos = kokkos

    @property
    def mtp_file_path(self) -> Path:
        """Path to the trained MTP potential file."""
        return self._mtp_file_path

    def interaction_commands(self, elements_string: str, with_uncertainty: bool = False) -> List[str]:
        """Return the MTP interaction commands.

        Without uncertainty it uses the plain 'mtp' pair_style; with uncertainty it uses 'mtp/extrapolation'
        and adds the extrapolation-grade fix.
        """
        pair_style = "mtp/extrapolation" if with_uncertainty else "mtp"
        commands = [
            f"pair_style {pair_style} {self._mtp_file_path}",
            "pair_coeff * *",
        ]
        if with_uncertainty:
            commands.append(f"fix mtp_grade all pair 1 {pair_style} extrapolation 1")
        return commands

    def dump_fields(self, with_uncertainty: bool = False) -> List[str]:
        """Return the per-atom fields written to the main dump."""
        fields = super().dump_fields(with_uncertainty=with_uncertainty)
        if with_uncertainty:
            fields = fields + ["f_mtp_grade"]
        return fields

    def uncertainty_field(self) -> str:
        """Return the per-atom uncertainty (extrapolation grade) column name."""
        return "f_mtp_grade"

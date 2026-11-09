"""Stillinger-Weber potential."""

from pathlib import Path
from typing import List

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential


class StillingerWeberPotential(LammpsPotential):
    """Stillinger-Weber potential."""

    calculation_type = "stillinger_weber"
    name = "SW"

    def __init__(self, sw_coefficients_file_path: Path):
        """Init method.

        Args:
            sw_coefficients_file_path: path to the Stillinger-Weber coefficient file.
        """
        self._sw_coefficients_file_path = sw_coefficients_file_path

    def interaction_commands(self, elements_string: str, with_uncertainty: bool = False) -> List[str]:
        """Return the pair_style and pair_coeff commands."""
        if with_uncertainty:
            raise ValueError("The Stillinger-Weber potential does not provide uncertainty.")
        return [
            "pair_style sw",
            f"pair_coeff * * {self._sw_coefficients_file_path} {elements_string}",
        ]

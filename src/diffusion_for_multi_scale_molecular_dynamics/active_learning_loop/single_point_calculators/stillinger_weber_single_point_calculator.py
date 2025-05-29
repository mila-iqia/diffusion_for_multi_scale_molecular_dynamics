from pathlib import Path
from typing import List

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_lammps_single_point_calculator import \
    BaseLAMMPSSinglePointCalculator  # noqa


class StillingerWeberSinglePointCalculator(BaseLAMMPSSinglePointCalculator):
    """Stillinger Weber Single Point Calculator."""

    def __init__(self, lammps_executable_path: Path, sw_coefficients_file_path: Path):
        """Init method."""
        super().__init__(lammps_executable_path)

        self._calculation_type = "stillinger_weber"
        self._sw_coefficients_file_path = sw_coefficients_file_path

    def _generate_pair_style_commands(self, elements_string: str) -> List[str]:
        commands = []
        commands.append("pair_style sw")
        commands.append(f"pair_coeff * * {self._sw_coefficients_file_path} {elements_string}")
        return commands

    def _generate_dump_commands(self, elements_string: str) -> List[str]:
        commands = []
        commands.append("dump 1 all yaml 1 dump.yaml id element x y z fx fy fz")
        commands.append(f"dump_modify 1 element {elements_string}")
        commands.append("dump_modify 1 thermo yes")
        commands.append("thermo 1")
        commands.append("thermo_style custom pe")

        return commands

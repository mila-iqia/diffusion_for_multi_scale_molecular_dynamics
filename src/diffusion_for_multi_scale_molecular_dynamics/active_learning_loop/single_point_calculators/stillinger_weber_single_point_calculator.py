from pathlib import Path

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_lammps_single_point_calculator import \
    BaseLAMMPSSinglePointCalculator  # noqa


class StillingerWeberSinglePointCalculator(BaseLAMMPSSinglePointCalculator):
    """Stillinger Weber Single Point Calculator."""

    def __init__(self, lammps_executable_path: Path, sw_coefficients_file_path: Path):
        """Init method."""
        super().__init__(lammps_executable_path)

        self._calculation_type = "stillinger_weber"
        self._sw_coefficients_file_path = sw_coefficients_file_path

    def _generate_pair_coeff_command(self, elements_string: str) -> str:
        return f"pair_coeff * * {self._sw_coefficients_file_path} {elements_string}"

    def _generate_pair_style_command(self) -> str:
        return "pair_style sw"

    def _generate_uncertainty_variable_string(self) -> str:
        return ""

from pathlib import Path

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.lammps_runner import \
    LammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.namespace import \
    UNCERTAINTY_FIELD
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_lammps_single_point_calculator import \
    BaseLAMMPSSinglePointCalculator  # noqa


class MappedFlareSinglePointCalculator(BaseLAMMPSSinglePointCalculator):
    """Mapped FLARE Single Point Calculator."""

    def __init__(self, lammps_runner: LammpsRunner, pair_coeff_file_path: Path, mapped_uncertainty_file_path: Path):
        """Init method."""
        super().__init__(lammps_runner)
        self._calculation_type = "mapped_flare"

        assert pair_coeff_file_path.is_file(), \
            f"The file '{pair_coeff_file_path}' does not exist. Review input."
        assert mapped_uncertainty_file_path.is_file(), \
            f"The file '{mapped_uncertainty_file_path}' does not exist. Review input."

        self._pair_coeff_path = pair_coeff_file_path
        self._map_unc_path = mapped_uncertainty_file_path

    def _generate_pair_coeff_command(self, elements_string: str) -> str:
        line1 = f"pair_coeff * * {self._pair_coeff_path}"
        line2 = f"compute unc all flare/std/atom {self._map_unc_path}"
        return line1 + '\n' + line2

    def _generate_pair_style_command(self) -> str:
        return "pair_style flare"

    def _generate_uncertainty_variable_string(self) -> str:
        return UNCERTAINTY_FIELD

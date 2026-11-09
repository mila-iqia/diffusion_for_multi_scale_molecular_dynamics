from typing import Any, AnyStr, Dict, Union

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.stillinger_weber import \
    StillingerWeberPotential
from diffusion_for_multi_scale_molecular_dynamics.oracle import \
    SW_COEFFICIENTS_DIR
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    BaseSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_single_point_calculator import \
    LammpsSinglePointCalculator


def instantiate_single_point_calculator(
        single_point_calculator_configuration: Dict[AnyStr, Any],
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
) -> BaseSinglePointCalculator:
    """Create a single point calculator.

    Args:
        single_point_calculator_configuration: input parameters that describe the calculator.
        lammps_runner: a runner that executes LAMMPS, injected into LAMMPS-based calculators.

    Returns:
        single_point_calculator: a single-point calculator.
    """
    calculator_name = single_point_calculator_configuration["name"]

    match calculator_name:

        case "stillinger_weber":
            sw_filename = single_point_calculator_configuration["sw_coeff_filename"]
            sw_coefficients_file_path = SW_COEFFICIENTS_DIR / sw_filename
            potential = StillingerWeberPotential(sw_coefficients_file_path=sw_coefficients_file_path)
            calculator = LammpsSinglePointCalculator(lammps_potential=potential, lammps_runner=lammps_runner)

        case _:
            raise NotImplementedError("Only stillinger weber is implemented at this time.")

    return calculator

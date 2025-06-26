from typing import Any, AnyStr, Dict

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.lammps_runner import \
    LammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_single_point_calculator import \
    BaseSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.stillinger_weber_single_point_calculator import \
    StillingerWeberSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.oracle import \
    SW_COEFFICIENTS_DIR


def instantiate_single_point_calculator(
        single_point_calculator_configuration: Dict[AnyStr, Any],
        lammps_runner: LammpsRunner,
) -> BaseSinglePointCalculator:
    """Create a single point calcculator.

    Args:
        single_point_calculator_config: input parameters that describe the calculator.
        lammps_runner: A LAMMPS runner, which may or may not be needed by the calculator.

    Returns:
        single_point_calculator: a single-point calculator.
    """
    calculator_name = single_point_calculator_configuration["name"]

    match calculator_name:

        case "stillinger_weber":
            sw_filename = single_point_calculator_configuration["sw_coeff_filename"]
            sw_coefficients_file_path = SW_COEFFICIENTS_DIR / sw_filename
            calculator = StillingerWeberSinglePointCalculator(lammps_runner=lammps_runner,
                                                              sw_coefficients_file_path=sw_coefficients_file_path)

        case _:
            raise NotImplementedError("Only stillinger weber is implemented at this time.")

    return calculator

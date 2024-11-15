from typing import Any, AnyStr, Dict, List

from diffusion_for_multi_scale_molecular_dynamics.oracle.energy_oracle import (
    EnergyOracle, OracleParameters)
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_energy_oracle import (
    LammpsEnergyOracle, LammpsOracleParameters)

ORACLE_PARAMETERS_BY_NAME = dict(lammps=LammpsOracleParameters)
ENERGY_ORACLE_BY_NAME = dict(lammps=LammpsEnergyOracle)


def create_energy_oracle_parameters(
    energy_oracle_dictionary: Dict[AnyStr, Any], elements: List[str]
) -> OracleParameters:
    """Create energy oracle parameters.

    Args:
        energy_oracle_dictionary : parsed configuration for the energy oracle.
        elements : list of unique elements.

    Returns:
        oracle_parameters: a configuration object for an energy oracle object.
    """
    name = energy_oracle_dictionary["name"]

    assert (
        name in ORACLE_PARAMETERS_BY_NAME.keys()
    ), f"Energy Oracle {name} is not implemented. Possible choices are {ORACLE_PARAMETERS_BY_NAME.keys()}"

    oracle_parameters = ORACLE_PARAMETERS_BY_NAME[name](
        **energy_oracle_dictionary, elements=elements
    )
    return oracle_parameters


def create_energy_oracle(oracle_parameters: OracleParameters) -> EnergyOracle:
    """Create an energy oracle.

    This is a factory method responsible for instantiating the energy oracle.
    """
    name = oracle_parameters.name
    assert (
        name in ENERGY_ORACLE_BY_NAME.keys()
    ), f"Energy Oracle {name} is not implemented. Possible choices are {ENERGY_ORACLE_BY_NAME.keys()}"

    oracle = ENERGY_ORACLE_BY_NAME[name](oracle_parameters)

    return oracle

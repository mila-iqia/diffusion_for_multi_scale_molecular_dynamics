"""Utility functions for data processing."""
import os
from typing import Any, AnyStr, Dict, List, Tuple

import yaml


def crop_lammps_yaml(lammps_dump: str, lammps_thermo: str, crop_step: int, inplace: bool = False) \
        -> Tuple[List[Dict[AnyStr, Any]], Dict[AnyStr, Any]]:
    """Remove the first steps of a LAMMPS run to remove structures near the starting point.

    Args:
        lammps_dump: path to LAMMPS output file as a yaml
        lammps_thermo: path to LAMMPS thermodynamic output file as a yaml
        crop_step: number of steps to remove
        inplace (optional): if True, overwrite the two LAMMPS file with a cropped version. If False, do not write.
            Defaults to False.

    Returns:
        cropped LAMMPS output file
        cropped LAMMPS thermodynamic output file
    """
    if not os.path.exists(lammps_dump):
        raise ValueError(f'{lammps_dump} does not exist. Please provide a valid LAMMPS dump file as yaml.')

    if not os.path.exists(lammps_thermo):
        raise ValueError(f'{lammps_thermo} does not exist. Please provide a valid LAMMPS thermo log file as yaml.')

    # get the atom information (positions and forces) from the LAMMPS 'dump' file
    with open(lammps_dump, 'r') as f:
        dump_yaml = yaml.safe_load_all(f)
        dump_yaml = [d for d in dump_yaml]  # generator to list
    # every MD iteration is saved as a separate document in the yaml file
    # prepare a dataframe to get all the data
    if crop_step >= len(dump_yaml):
        raise ValueError(f"Trying to remove {crop_step} steps in a run of {len(dump_yaml)} steps.")
    dump_yaml = dump_yaml[crop_step:]

    # get the total energy from the LAMMPS thermodynamic output
    with open(lammps_thermo, 'r') as f:
        thermo_yaml = yaml.safe_load(f)
    thermo_yaml['data'] = thermo_yaml['data'][crop_step:]

    if inplace:
        with open("test_yaml.yaml", "w") as f:
            yaml.dump_all(dump_yaml, f, explicit_start=True)
        with open("test_thermo.yaml", "w") as f:
            yaml.dump(thermo_yaml, f)

    return dump_yaml, thermo_yaml

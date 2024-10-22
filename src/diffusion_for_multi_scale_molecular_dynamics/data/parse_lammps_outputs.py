import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from yaml import CLoader


def parse_lammps_output(
    lammps_dump: str, lammps_thermo_log: str, output_name: Optional[str] = None
) -> pd.DataFrame:
    """Parse a LAMMPS output file and save in a .csv format.

    Args:
        lammps_dump: LAMMPS output file
        lammps_thermo_log: LAMMPS thermodynamic variables output file
        output_name (optional): name of parsed output written by the script. If none, do not write data to disk.
            Defaults to None.

    Returns:
        data in a dataframe
    """
    if not os.path.exists(lammps_dump):
        raise ValueError(
            f"{lammps_dump} does not exist. Please provide a valid LAMMPS dump file as yaml."
        )

    if not os.path.exists(lammps_thermo_log):
        raise ValueError(
            f"{lammps_thermo_log} does not exist. Please provide a valid LAMMPS thermo log file as yaml."
        )

    # get the atom information (positions and forces) from the LAMMPS 'dump' file
    pd_data = parse_lammps_dump(lammps_dump)

    # get the total energy from the LAMMPS second output
    thermo_log_data_dictionary = parse_lammps_thermo_log(lammps_thermo_log)
    pd_data.update(thermo_log_data_dictionary)

    if output_name is not None and not output_name.endswith(".parquet"):
        output_name += ".parquet"

    df = pd.DataFrame(pd_data)

    if output_name is not None:
        df.to_parquet(output_name, engine="pyarrow", index=False)

    return df


def parse_lammps_dump(lammps_dump: str) -> Dict[str, Any]:
    """Parse lammps dump.

    We make the assumption that the content of the dump file is for 3D data.

    Args:
        lammps_dump : path to lammps dump file, in yaml format.

    Returns:
        data: a dictionary with all the relevant data.
    """
    expected_keywords = ["id", "type", "x", "y", "z", "fx", "fy", "fz"]
    datatypes = 2 * [np.int64] + 6 * [np.float64]

    pd_data = defaultdict(list)
    with open(lammps_dump, "r") as stream:
        dump_yaml = yaml.load_all(stream, Loader=CLoader)

        for doc in dump_yaml:  # loop over MD steps
            pd_data["box"].append(np.array(doc["box"])[:, 1])

            assert doc["keywords"] == expected_keywords
            data = np.array(
                doc["data"]
            ).transpose()  # convert to numpy so that we can easily slice
            for keyword, datatype, data_row in zip(expected_keywords, datatypes, data):
                pd_data[keyword].append(data_row.astype(datatype))
    return pd_data


def parse_lammps_thermo_log(lammps_thermo_log: str) -> Dict[str, List[float]]:
    """Parse LAMMPS thermo log.

    Args:
        lammps_thermo_log : path to the lammps thermo log.

    Returns:
        parsed_data: the data from the log, parsed in a dictionary.
    """
    data_dict = defaultdict(list)
    optional_keywords = {"Press": "pressure", "Temp": "temperature"}
    optional_indices = dict()

    with open(lammps_thermo_log, "r") as f:
        log_yaml = yaml.safe_load(f)
        kin_idx = log_yaml["keywords"].index("KinEng")
        pot_idx = log_yaml["keywords"].index("PotEng")

        # For optional keys, impose better names than the LAMMPS keys.
        for yaml_key, long_name in optional_keywords.items():
            if yaml_key in log_yaml["keywords"]:
                idx = log_yaml["keywords"].index(yaml_key)
                optional_indices[long_name] = idx

        for record in log_yaml["data"]:
            potential_energy = record[pot_idx]
            kinetic_energy = record[kin_idx]
            data_dict["potential_energy"].append(potential_energy)
            data_dict["kinetic_energy"].append(kinetic_energy)
            data_dict["energy"].append(potential_energy + kinetic_energy)

            for long_name, idx in optional_indices.items():
                data_dict[long_name].append(record[idx])

    return data_dict


def main():
    """Parse LAMMPS files and output a single parquet file."""
    parser = argparse.ArgumentParser(
        description="Convert LAMMPS outputs in parquet file compatible with a dataloader."
    )
    parser.add_argument(
        "--dump_file", type=str, help="LAMMPS dump file in yaml format."
    )
    parser.add_argument(
        "--thermo_file", type=str, help="LAMMPS thermo output file in yaml format."
    )
    parser.add_argument("--output_name", type=str, help="Output name")
    args = parser.parse_args()

    parse_lammps_output(args.dump_file, args.thermo_file, args.output_name)


if __name__ == "__main__":
    main()

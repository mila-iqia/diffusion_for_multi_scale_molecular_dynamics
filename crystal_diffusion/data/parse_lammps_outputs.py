import argparse
import os
from collections import defaultdict

import pandas as pd
import yaml


def parse_lammps_output(lammps_dump: str, lammps_thermo_log: str, output_name: str):
    """Parse a LAMMPS output file and save in a .csv format.

    Args:
        lammps_dump: LAMMPS output file
        lammps_thermo_log: LAMMPS thermodynamic variables output file
        output_name: name of parsed output written by the script
    """
    if not os.path.exists(lammps_dump):
        raise ValueError(f'{lammps_dump} does not exist. Please provide a valid LAMMPS dump file as yaml.')

    if not os.path.exists(lammps_thermo_log):
        raise ValueError(f'{lammps_thermo_log} does not exist. Please provide a valid LAMMPS thermo log file as yaml.')

    # get the atom information (positions and forces) from the LAMMPS 'dump' file
    with open(lammps_dump, 'r') as f:
        dump_yaml = yaml.safe_load_all(f)
        # every MD iteration is saved as a separate document in the yaml file
        # prepare a dataframe to get all the data
        pd_data = defaultdict(list)
        for doc in dump_yaml:  # loop over MD steps
            if 'id' not in doc['keywords']:  # sanity check
                raise ValueError('id should be in LAMMPS dump file')
            atoms_info = defaultdict(list) # store information on atoms positions and forces here

            for data in doc['data']:  # loop over the atoms to get their positions and forces
                for key, v in zip(doc['keywords'], data):
                    if key not in ['id', 'type', 'x', 'y', 'z', 'fx', 'fy', 'fz']:
                        continue
                    else:
                        atoms_info[key].append(v)  # get positions or forces
            # add the information about that MD step to the dataframe
            for k, v in atoms_info.items():  # k should be id, type, x, y, z, fx, fy, fz
                pd_data[k].append(v)

    # get the total energy from the LAMMPS second output
    with open(lammps_thermo_log, 'r') as f:
        log_yaml = yaml.safe_load(f)
        kin_idx = log_yaml['keywords'].index('KinEng')
        pot_idx = log_yaml['keywords'].index('PotEng')
        pd_data['energy'] = [x[kin_idx] + x[pot_idx] for x in log_yaml['data']]

    if not output_name.endswith('.parquet'):
        output_name += '.parquet'

    pd.DataFrame(pd_data).to_parquet(output_name, engine='pyarrow', index=False)


def main():
    """Main script to parse LAMMPS files and output a single parquet file."""
    parser = argparse.ArgumentParser(description="Convert LAMMPS outputs in parquet file compatible with a dataloader.")
    parser.add_argument("--dump_file", type=str, help="LAMMPS dump file in yaml format.")
    parser.add_argument("--thermo_file", type=str, help="LAMMPS thermo output file in yaml format.")
    parser.add_argument("--output_name", type=str, help="Output name")
    args = parser.parse_args()

    parse_lammps_output(args.dump_file, args.thermo_file, args.output_name)


if __name__ == '__main__':
    main()

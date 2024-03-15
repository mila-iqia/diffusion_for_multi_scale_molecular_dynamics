"""Convert MTP predictions to OVITO readable file.

Draw the MD simulation with the MaxVol values. Some tweaking is required in the OVITO UI. See README_OVITO.md for more
information.
"""
import argparse
import os

import numpy as np
import pandas as pd
import yaml


def main():
    """Read MTP output files and convert to xyz format readable by OVITO."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", help="MTP prediction files. Should contain position and MaxVol gamma")
    parser.add_argument("--lammps_output", help="MTP prediction files. Should contain position and MaxVol gamma")
    parser.add_argument("--output_name", help="Name of the output file that can be loaded by OVITO. ")
    args = parser.parse_args()

    assert os.path.exists(args.lammps_output), f"LAMMPS out file {args.lammps_output} does not exist."

    lattice = get_lattice_from_lammps(args.lammps_output)

    assert os.path.exists(args.prediction_file), f"Provided prediction file {args.prediction_file} does not exist."

    mtp_predictions_to_ovito(args.prediction_file, lattice, args.output_name)


def get_lattice_from_lammps(lammps_output_file: str) -> np.ndarray:
    """Read periodic bounding box from LAMMPS output file.

    Args:
        lammps_output_file: lammps output file (dump)

    Returns:
        lattice: 3x3 array with lattice coordinates
    """
    with (open(lammps_output_file, 'r') as f):
        l_yaml = yaml.safe_load_all(f)
        for d in l_yaml:  # loop over LAMMPS outputs to get the MD box - we only need the first step
            # lattice in yaml is 3 x 2 [0, x_lim]
            # we assume a rectangular lattice for now with the 2nd coordinates as the lattice vectors
            print(d)
            lattice = np.zeros((3, 3))
            for i, x in enumerate(d['box']):
                lattice[i, i] = x[1]
            break
    return lattice


def mtp_predictions_to_ovito(pred_file: str, lattice: np.ndarray, output_name: str):
    """Convert output csv to ovito readable file.

    Args:
        pred_file: output csv to convert
        lattice: lattice parameters in a 3x3 numpy array
        output_name: name of resulting file. An .xyz extension is added if not already in the name.
    """
    lattice = list(map(str, lattice.flatten()))  # flatten and convert to string for formatting
    lattice_str = 'Lattice=\"' + " ".join(lattice) + '\" Origin=\"0 0 0\" pbc=\"T T T\"'
    df = pd.read_csv(pred_file)  # read the predictions
    xyz_file = ""  # output will be written in file
    for struct in sorted(df['structure_index'].unique()):  # iterate over LAMMPS steps
        xyz_values = df.loc[df['structure_index'] == struct, ['x', 'y', 'z']].to_numpy()
        gamma_values = df.loc[df['structure_index'] == struct, 'nbh_grades'].to_numpy()  # nbh_grade
        n_atom = xyz_values.shape[0]
        frame_txt = f"{n_atom}\n"
        frame_txt += (lattice_str + ' Properties=pos:R:3:MaxVolGamma:R:1\n')
        # here, we can add properties to filter out atoms or identify some of them
        for i in range(n_atom):
            frame_txt += f"{' '.join(map(str, xyz_values[i, :]))} {gamma_values[i]}\n"
        xyz_file += frame_txt

    if not output_name.endswith(".xyz"):
        output_name += ".xyz"

    with open(output_name, 'w') as f:
        f.write(xyz_file)


if __name__ == '__main__':
    main()

"""Read and crop LAMMPS outputs."""
import argparse
import os

import yaml

from crystal_diffusion.data.utils import crop_lammps_yaml


def main():
    """Read LAMMPS outputs from arguments and crops."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--lammps_yaml', help='path to LAMMPS yaml file', required=True)
    parser.add_argument('--lammps_thermo', help='path to LAMMPS thermo output', required=True)
    parser.add_argument('--crop', help='number of steps to remove at the start of the run', required=True)
    parser.add_argument('--output_dir', help='path to folder where outputs will be saved', required=True)
    args = parser.parse_args()

    lammps_yaml = args.lammps_yaml
    lammps_thermo_yaml = args.lammps_thermo

    lammps_yaml, lammps_thermo_yaml = crop_lammps_yaml(lammps_yaml, lammps_thermo_yaml, args.crop, inplace=False)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'lammps_dump.yaml'), 'w') as f:
        yaml.dump_all(lammps_yaml, f, explicit_start=True)
    with open(os.path.join(args.output_dir, 'lammps_thermo.yaml'), 'w') as f:
        yaml.dump(lammps_thermo_yaml, f)


if __name__ == '__main__':
    main()

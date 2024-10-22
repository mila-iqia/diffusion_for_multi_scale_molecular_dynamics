"""Read and crop LAMMPS outputs."""
import argparse
import logging
import os

import yaml

from diffusion_for_multi_scale_molecular_dynamics.data.utils import \
    crop_lammps_yaml
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    setup_analysis_logger

logger = logging.getLogger(__name__)


def main():
    """Read LAMMPS outputs from arguments and crops."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--lammps_yaml', help='path to LAMMPS yaml file', required=True)
    parser.add_argument('--lammps_thermo', help='path to LAMMPS thermo output', required=True)
    parser.add_argument('--crop', type=int, help='number of steps to remove at the start of the run', required=True)
    parser.add_argument('--output_dir', help='path to folder where outputs will be saved', required=True)
    args = parser.parse_args()

    lammps_yaml = args.lammps_yaml
    lammps_thermo_yaml = args.lammps_thermo

    logger.info(f"Cropping LAMMPS files {lammps_yaml} and {lammps_thermo_yaml}...")
    lammps_yaml, lammps_thermo_yaml = crop_lammps_yaml(lammps_yaml, lammps_thermo_yaml, args.crop, inplace=False)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Dumping cropped dump data to file...")
    with open(os.path.join(args.output_dir, 'lammps_dump.yaml'), 'w') as f:
        yaml.dump_all(lammps_yaml, f, explicit_start=True, sort_keys=False, default_flow_style=None, width=1000)
    logger.info("Dumping cropped thermo data to file...")
    with open(os.path.join(args.output_dir, 'lammps_thermo.yaml'), 'w') as f:
        yaml.dump(lammps_thermo_yaml, f, sort_keys=False, default_flow_style=None, width=1000)


setup_analysis_logger()
if __name__ == '__main__':
    main()

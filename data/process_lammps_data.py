"""Create the processed data."""
import argparse
import tempfile

from crystal_diffusion.data.diffusion.data_loader import (
    LammpsForDiffusionDataModule, LammpsLoaderParameters)
from crystal_diffusion.utils.logging_utils import setup_analysis_logger


def main():
    """Read LAMMPS directories from arguments and process data."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to a LAMMPS data set', required=True)
    parser.add_argument('--processed_datadir', help='path to the processed data directory', required=True)
    parser.add_argument('--max_atom', help='maximum number of atoms', required=True)
    args = parser.parse_args()

    lammps_run_dir = args.data
    processed_dataset_dir = args.processed_datadir
    data_params = LammpsLoaderParameters(batch_size=128, num_workers=0, max_atom=args.max_atom)

    with tempfile.TemporaryDirectory() as tmp_work_dir:
        data_module = LammpsForDiffusionDataModule(lammps_run_dir=lammps_run_dir,
                                                   processed_dataset_dir=processed_dataset_dir,
                                                   hyper_params=data_params,
                                                   working_cache_dir=tmp_work_dir)
        data_module.setup()


setup_analysis_logger()
if __name__ == '__main__':
    main()

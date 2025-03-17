"""Create the processed data."""
import argparse
import logging
import tempfile

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import (
    LammpsDataModuleParameters, LammpsForDiffusionDataModule)
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    setup_analysis_logger
from diffusion_for_multi_scale_molecular_dynamics.utils.main_utils import \
    _get_hyperparameters

logger = logging.getLogger(__name__)


def main():
    """Read LAMMPS directories from arguments and process data."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to a LAMMPS data set', required=True)
    parser.add_argument('--processed_datadir', help='path to the processed data directory', required=True)
    parser.add_argument("--config", help="config file with dataloader hyper-parameters, such as "
                                         "batch_size, elements, ... -  in yaml format")
    args = parser.parse_args()

    lammps_run_dir = args.data
    processed_dataset_dir = args.processed_datadir
    hyper_params = _get_hyperparameters(config_file_path=args.config)

    logger.info("Starting process_lammps_data.py script with arguments")
    logger.info(f"   --data : {args.data}")
    logger.info(f"   --processed_datadir : {args.processed_datadir}")
    logger.info(f"   --config: {args.config}")

    data_params = LammpsDataModuleParameters(**hyper_params)

    with tempfile.TemporaryDirectory() as tmp_work_dir:
        data_module = LammpsForDiffusionDataModule(lammps_run_dir=lammps_run_dir,
                                                   processed_dataset_dir=processed_dataset_dir,
                                                   hyper_params=data_params,
                                                   working_cache_dir=tmp_work_dir)
        data_module.setup()


setup_analysis_logger()
if __name__ == '__main__':
    main()

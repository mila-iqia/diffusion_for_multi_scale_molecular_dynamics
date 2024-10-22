import glob
import logging
import os
from typing import Tuple

import pandas as pd

from diffusion_for_multi_scale_molecular_dynamics import DATA_DIR
from diffusion_for_multi_scale_molecular_dynamics.data.parse_lammps_outputs import \
    parse_lammps_thermo_log
from experiments import EXPERIMENT_ANALYSIS_DIR

logger = logging.getLogger(__name__)


def get_thermo_dataset(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get thermo dataset.

    This function fetches the training and validation thermo properties,
    assuming that the data is located in various train_runs and validation_runs
    in the DATA_DIR directory.

    Data will be cached in pandas pickles for quick re-use.

    Args:
        dataset_name : name of the dataset, which should match the directory where the
            dataset is located in the DATA_DIR folder.

    Returns:
        train_df, valid_df: training data and validation data, respectively.

    """
    lammps_dataset_dir = DATA_DIR.joinpath(dataset_name)
    assert lammps_dataset_dir.is_dir(), \
        f"The folder {lammps_dataset_dir} does not exist! Data must be present to execute this function."

    cache_dir = EXPERIMENT_ANALYSIS_DIR.joinpath(f"cache/{dataset_name}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    list_train_df = []
    list_valid_df = []

    logging.info("Parsing the thermo logs")
    run_directories = glob.glob(str(lammps_dataset_dir.joinpath('*_run_*')))

    for run_directory in run_directories:
        basename = os.path.basename(run_directory)
        pickle_path = cache_dir.joinpath(f"{basename}.pkl")
        if os.path.isfile(pickle_path):
            logging.info(f"Pickle file {pickle_path} exists. Reading in...")
        else:
            logging.info(f"Pickle file {pickle_path} does not exist: creating...")
            lammps_thermo_log = lammps_dataset_dir.joinpath(f"{basename}/lammps_thermo.yaml")
            df = pd.DataFrame(parse_lammps_thermo_log(lammps_thermo_log))
            df.to_pickle(pickle_path)
            logging.info("Done creating pickle file")

        df = pd.read_pickle(pickle_path)
        if 'train' in basename:
            list_train_df.append(df)
        else:
            list_valid_df.append(df)

    train_df = pd.concat(list_train_df)
    valid_df = pd.concat(list_valid_df)

    return train_df, valid_df

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from crystal_diffusion.analysis import (ANALYSIS_DIR, PLEASANT_FIG_SIZE,
                                        PLOT_STYLE_PATH)

plt.style.use(PLOT_STYLE_PATH)

dataset_name = 'si_diffusion_1x1x1'

output_directory = Path(__file__).parent.joinpath("si_diffusion_1x1x1_output")
sample_pickle_path = output_directory.joinpath('energy_samples.pkl')

training_data_cache_dir = ANALYSIS_DIR.joinpath("cache/si_diffusion_1x1x1")


if __name__ == '__main__':
    sample_df = pd.read_pickle(sample_pickle_path)

    list_train_df = []
    for train_pickle_path in glob.glob(str(training_data_cache_dir.joinpath('*.pkl'))):
        basename = os.path.basename(train_pickle_path)
        df = pd.read_pickle(train_pickle_path)
        if 'train' in basename:
            list_train_df.append(df)
    train_df = pd.concat(list_train_df)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)

    fig.suptitle(f'Comparing Training and Sampling Energy Distributions\n Dataset = {dataset_name}')

    ax1 = fig.add_subplot(111)

    common_params = dict(density=True, bins=50, histtype="stepfilled", alpha=0.25)
    ax1.hist(sample_df['energy'], **common_params, label=f'Samples (count = {len(sample_df)})', color='red')
    ax1.hist(train_df['energy'], **common_params, label=f'Training Data (count = {len(train_df)})', color='green')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Density')
    ax1.legend(loc=0)
    plt.show()

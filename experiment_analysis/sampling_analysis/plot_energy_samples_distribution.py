from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crystal_diffusion import ANALYSIS_RESULTS_DIR
from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from experiment_analysis.analysis_utils import get_thermo_dataset

plt.style.use(PLOT_STYLE_PATH)

dataset_name = 'si_diffusion_2x2x2'


if dataset_name == 'si_diffusion_1x1x1':
    dataset_name_for_data_fetching = 'si_diffusion_1x1x1'
    upper_limit_factor = 0.1
elif dataset_name == 'si_diffusion_2x2x2':
    dataset_name_for_data_fetching = 'si_diffusion_small'
    upper_limit_factor = 5.

sample_pickle_path = Path(__file__).parent.joinpath(f'{dataset_name}_energy_samples.pkl')


if __name__ == '__main__':
    sample_df = pd.read_pickle(sample_pickle_path)

    train_df, _ = get_thermo_dataset(dataset_name_for_data_fetching)

    train_energies = train_df['energy'].values
    sample_energies = sample_df['energy'].values

    delta_e = train_energies.max() - train_energies.min()
    emin = train_energies.min() - 0.1 * delta_e
    emax = train_energies.max() + upper_limit_factor * delta_e

    number_of_samples_in_range = np.logical_and(sample_energies >= emin, sample_energies <= emax).sum()
    bins = np.linspace(emin, emax, 101)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)

    fig.suptitle(f'Comparing Training and Sampling Energy Distributions\n Dataset = {dataset_name}')

    ax1 = fig.add_subplot(111)

    common_params = dict(density=True, bins=bins, histtype="stepfilled", alpha=0.25)
    ax1.hist(sample_energies, **common_params,
             label=f'Samples (total count = {len(sample_df)}, in range = {number_of_samples_in_range})', color='red')
    ax1.hist(train_energies, **common_params, label=f'Training Data (count = {len(train_df)})', color='green')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Density')
    ax1.legend(loc=0)

    fig.tight_layout()
    fig.savefig(ANALYSIS_RESULTS_DIR.joinpath(f"{dataset_name}_sample_energy_distribution.png"))

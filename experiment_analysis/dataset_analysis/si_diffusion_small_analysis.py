"""Si diffusion dataset analysis.

Si datasets were generated in early april, where 'small' means configurations containing 64 atoms (a 2x2x2 supercell),
and 1x1x1 means configurations containing only 8 atoms. These datasets were processed into parquet files and used
to run experiments.

In this script, the energy vs. temperature relationships of these datasets will be analysed starting from the
thermo logs.

It is still early days in the project, so the starting point format for these kinds of analyses is still in flux.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from crystal_diffusion import ANALYSIS_RESULTS_DIR, DATA_DIR
from crystal_diffusion.analysis import (ANALYSIS_DIR, PLEASANT_FIG_SIZE,
                                        PLOT_STYLE_PATH)
from crystal_diffusion.utils.logging_utils import setup_analysis_logger
from experiment_analysis.analysis_utils import get_thermo_dataset

plt.style.use(PLOT_STYLE_PATH)

logger = logging.getLogger(__name__)

#  Taken from this table: http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table.html
kelvin_in_ev = 0.0000861705

dataset_name = 'si_diffusion_1x1x1'

if dataset_name == 'si_diffusion_1x1x1':
    number_of_atoms = 8
elif dataset_name == 'si_diffusion_small':
    number_of_atoms = 64

lammps_dataset_dir = DATA_DIR.joinpath(dataset_name)

cache_dir = ANALYSIS_DIR.joinpath(f"cache/{dataset_name}")
cache_dir.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    setup_analysis_logger()

    logging.info(f"Starting {dataset_name} analysis")

    train_df, valid_df = get_thermo_dataset(dataset_name)

    temperatures = train_df['temperature'].values  # Kelvin
    mu_temp = np.mean(temperatures)
    kB_T = mu_temp * kelvin_in_ev
    sigma_temp = np.std(temperatures)

    list_t = np.linspace(100, 500, 1001)
    list_pt = scipy.stats.norm(loc=mu_temp, scale=sigma_temp).pdf(list_t)

    beta = 1. / kB_T

    relative_energies = train_df['energy'].values
    mu_e = np.mean(relative_energies)
    e0 = mu_e - 3. * number_of_atoms * kB_T

    expected_sigma_e = np.sqrt((3. * number_of_atoms) / beta**2)  # in eV
    computed_sigma_e = np.std(relative_energies)
    std_relative_error = (expected_sigma_e - computed_sigma_e) / expected_sigma_e

    list_de = np.linspace(mu_e - 5 * computed_sigma_e, mu_e + 5 * computed_sigma_e, 1001)
    list_pe = scipy.stats.norm(loc=mu_e, scale=computed_sigma_e).pdf(list_de)

    fig1 = plt.figure(figsize=PLEASANT_FIG_SIZE)

    fig1.suptitle(f'Distributions for dataset {dataset_name}')

    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)

    common_params = dict(density=True, bins=50, histtype="stepfilled", alpha=0.25)

    ax1.set_title('Energy distribution')
    ax1.hist(train_df['energy'], **common_params, label='Train', color='green')
    ax1.hist(valid_df['energy'], **common_params, label='Valid', color='yellow')
    ax1.vlines(e0, ymin=0, ymax=1, color='k', linestyle='--', label=f'$E_0$ = {e0:5.3f} eV')
    ax1.vlines(e0 + 3 * number_of_atoms * kB_T, ymin=0, ymax=1, color='r',
               linestyle='--', label=f'$E_0 + 3 N k_B T$ = {np.mean(relative_energies):5.3f} eV')

    label = f'Normal Fit: $\\sigma_{{emp}}$ = {computed_sigma_e:5.3f} eV, $\\sigma_{{th}}$ = {expected_sigma_e:5.3f} eV'
    ax1.plot(list_de, list_pe, '-', color='blue', label=label)

    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Density')
    ax1.legend(loc=0)

    ax2.set_title('Temperature Distribution')
    ax2.hist(train_df['temperature'], **common_params, label='Train', color='green')
    ax2.hist(valid_df['temperature'], **common_params, label='Valid', color='yellow')
    label = f'Normal Fit: $\\sigma_T$ = {sigma_temp:5.1f} K, $k_B \\sigma_T$ = {kelvin_in_ev * sigma_temp:5.4f} eV'
    ax2.plot(list_t, list_pt, '-', color='blue', label=label)
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Density')
    ax2.legend(loc=0)

    fig1.tight_layout()
    fig1.savefig(ANALYSIS_RESULTS_DIR.joinpath(f"{dataset_name}_distribution_analysis.png"))

    common_params = dict(density=True, bins=50, histtype="stepfilled", alpha=0.25)
    fig2 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    ax3 = fig2.add_subplot(121)
    ax4 = fig2.add_subplot(122)
    ax3.set_title('Energy distribution')
    ax4.set_title('Average Energy')

    ax3.set_xlabel('$E- E_0$ (eV)')
    ax3.set_ylabel('Density')

    ax4.set_xlabel('$k_B T$ (eV)')
    ax4.set_ylabel('Energy (eV)')

    df = train_df[['temperature', 'energy']]
    temperature_bins = pd.cut(df['temperature'], bins=5)

    list_kt = []
    list_kt_err = []
    list_de = []
    list_de_err = []
    for temp_bin, group_df in df.groupby(temperature_bins):
        relative_energies = group_df['energy'].values - e0
        kB_temperatures = kelvin_in_ev * group_df['temperature'].values
        ax3.hist(relative_energies, **common_params, label=f'$T \\in$ {temp_bin} K')
        list_kt.append(np.mean(kB_temperatures))
        list_kt_err.append(np.std(kB_temperatures))
        list_de.append(np.mean(relative_energies))
        list_de_err.append(np.std(relative_energies))

    ax4.errorbar(list_kt, list_de, xerr=list_kt_err, yerr=list_de_err, fmt='o', label='Data')

    list_kt_fit = np.linspace(0.01, 0.04, 101)
    list_de_ideal_fit = 3. * number_of_atoms * list_kt_fit
    ax4.plot(list_kt_fit, list_de_ideal_fit, 'r-', label='$E-E_0 = 3 N k_B T$')

    fit = np.polyfit(list_kt, list_de, deg=1)

    m = fit[0]
    b = fit[1]
    list_de_fit = np.poly1d(fit)(list_kt_fit)
    ax4.plot(list_kt_fit, list_de_fit, 'g-', label=f'$E-E_0 = m k_B T + b$\n$m = {m:4.2f}, b={b:4.2f}$ eV')

    ax3.legend(loc=0)
    ax4.legend(loc=0)
    fig2.tight_layout()
    fig2.savefig(ANALYSIS_RESULTS_DIR.joinpath(f"{dataset_name}_temperature_bins_analysis.png"))

    fig3 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    ax5 = fig3.add_subplot(111)
    ax5.hist2d(kelvin_in_ev * df['temperature'], df['energy'], bins=100)
    ax5.set_xlabel('$k_B T$ (eV)')
    ax5.set_ylabel('$E$ (eV)')

    list_kb_T = kelvin_in_ev * np.linspace(100, 500, 101)
    list_e = e0 + 3 * number_of_atoms * list_kb_T
    ax5.plot(list_kb_T, list_e, 'r-', label='$E = E_0 + 3 N k_B T$')

    ax5.legend(loc=0)

    fig3.tight_layout()
    fig3.savefig(ANALYSIS_RESULTS_DIR.joinpath(f"{dataset_name}_energy_temperature_2D_histogram.png"))

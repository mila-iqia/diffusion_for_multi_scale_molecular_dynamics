import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_single_point_calculator import \
    SinglePointCalculation  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.flare_single_point_calculator import \
    FlareSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.utils import \
    compute_errors_and_uncertainties
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import (
    FlareConfiguration, FlareTrainer)
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)

plt.style.use(PLOT_STYLE_PATH)


amorphous_exp_dir = Path("/Users/brunorousseau/courtois/july26/active_learning/amorphous_silicon/excise_and_repaint/")
oracle_path = amorphous_exp_dir / "output/run1/campaign_1/round_1/oracle"

# We assume that FLARE models have been pre-trained and are available here.
data_dir = TOP_DIR / "experiments/active_learning/pretraining_flare/data/"

analysis_dir = TOP_DIR / "analysis_and_sanity_checks/analyse_FLARE"

images_dir = analysis_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

sigma = 1.0
sigma_s = 1.0
sigma_e = 0.1
element_list = ['Si']

list_sigma_f = [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]


if __name__ == "__main__":

    with open(data_dir / "train_labelled_structures.pkl", "rb") as fd:
        list_train_labelled_structures = pickle.load(fd)

    train_labelled_structure = list_train_labelled_structures[0]

    df = pd.read_pickle(oracle_path / "oracle_single_point_calculations.pkl")

    list_repaint_labelled_structures = []
    for idx, row in df.iterrows():
        structure = row['structure']
        forces = np.array(structure.site_properties['forces'])

        labelled_structure = SinglePointCalculation(
            calculation_type=row['calculation_type'],
            structure=structure,
            forces=forces,
            energy=row['energy'])
        list_repaint_labelled_structures.append(labelled_structure)

    repaint_labelled_structure = list_repaint_labelled_structures[0]

    list_ax_titles = ["Configuration from Training Set", "Repainted Configuration"]

    list_df = []
    for labelled_structure in [train_labelled_structure, repaint_labelled_structure]:
        list_rows = []
        for number_of_envs in tqdm(np.arange(1, 64), "ENVS"):
            active_environment_indices = list(np.arange(number_of_envs))
            for sigma_f in list_sigma_f:
                flare_configuration = FlareConfiguration(
                    cutoff=5.0,
                    elements=element_list,
                    n_radial=12,
                    lmax=3,
                    initial_sigma=sigma,
                    initial_sigma_e=sigma_e,
                    initial_sigma_f=sigma_f,
                    initial_sigma_s=sigma_s,
                    variance_type="local",
                )
                flare_trainer = FlareTrainer(flare_configuration)

                flare_trainer.add_labelled_structure(single_point_calculation=labelled_structure,
                                                     active_environment_indices=active_environment_indices)
                flare_calculator = FlareSinglePointCalculator(sgp_model=flare_trainer.sgp_model)
                results = compute_errors_and_uncertainties(flare_calculator, [labelled_structure])

                row = dict(sigma_e=sigma_e,
                           sigma_f=sigma_f,
                           number_of_envs=number_of_envs,
                           mean_force_rmse=results['mean_force_rmse'],
                           energy_rmse=results['energy_rmse'])
                list_rows.append(row)

        df = pd.DataFrame.from_records(list_rows)
        list_df.append(df)

    figsize = (PLEASANT_FIG_SIZE[0], 0.75 * PLEASANT_FIG_SIZE[1])
    fig = plt.figure(figsize=figsize)
    fig.suptitle("FLARE Trained and Evaluated on a Single Configuration\n"
                 rf"$\sigma$ = {sigma}, $\sigma_e$ = {sigma_e}")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    list_ax = [ax1, ax2]

    for title, df, ax in zip(list_ax_titles, list_df, list_ax):
        ax.set_title(title)
        lw = 4
        for sigma_f, group_df in df.groupby('sigma_f'):
            sorted = group_df.sort_values(by='number_of_envs')
            number_of_envs = sorted['number_of_envs'].values
            mean_force_rmse = sorted['mean_force_rmse'].values
            energy_rmse = sorted['energy_rmse'].values
            ax.semilogy(number_of_envs, mean_force_rmse, '-', lw=lw, label=rf"$\sigma_f$={sigma_f}")
            lw = 0.9 * lw

        ax.legend(loc=0, fontsize=10)
        ax.set_xlim(0, 65)
        ax.set_ylim(1e-3, 10)
        ax.set_xlabel('Number of Envs added to Sparse Set')
        ax.set_ylabel('Force RMSE')

    fig.tight_layout()
    fig.savefig(images_dir / "flare_error_on_single_structure.png")

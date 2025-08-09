from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.analysis.saddle_energy_extraction import \
    extract_all_saddle_energies
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)

plt.style.use(PLOT_STYLE_PATH)


top_dir = Path("/Users/brunorousseau/courtois/july26/active_learning/amorphous_silicon")

# experiment = "excise_and_repaint"
experiment = "excise_and_repaint_3x3x3"
# experiment = "noop"
# experiment = "excise_and_noop"
# experiment = "excise_and_random"

top_experiment_directory = top_dir / experiment / "output"

if __name__ == "__main__":

    df = extract_all_saddle_energies(top_experiment_directory)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f"Saddle Point Energy From Active Learning\n Algorithm : {experiment}")
    ax1 = fig.add_subplot(111)

    for run_id, group_df in df.groupby(by="run_id"):
        list_thresholds = group_df['threshold']
        list_saddle_energies = group_df['saddle_energy']
        ax1.semilogx(list_thresholds, list_saddle_energies, "o-", ms=5, label=f"run {run_id}")

    list_thresholds = []
    list_mean = []
    list_std = []
    for threshold, group_df in df.groupby(by="threshold"):
        list_thresholds.append(threshold)
        saddle_energies = group_df['saddle_energy']
        list_mean.append(np.nanmean(saddle_energies))
        list_std.append(np.nanstd(saddle_energies))

    list_mean = np.array(list_mean)
    list_std = np.array(list_std)
    list_thresholds = np.array(list_thresholds)

    ax1.fill_between(
        list_thresholds,
        y1=list_mean - list_std,
        y2=list_mean + list_std,
        alpha=0.25,
        color="blue",
        label=r"$\pm$ One Standard Deviation",
    )

    ax1.set_ylabel("Saddle Point Energy (eV)")
    ax1.set_xlabel("Uncertainty Threshold (unitless)")

    ax1.legend(loc=3)
    fig.tight_layout()

    plt.show()

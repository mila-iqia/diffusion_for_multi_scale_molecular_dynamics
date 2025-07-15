import glob

import numpy as np
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.artn.artn_outputs import \
    get_saddle_energy
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)

plt.style.use(PLOT_STYLE_PATH)


top_dir = TOP_DIR / "experiments/active_learning/amorphous_silicon"
experiment_dir = top_dir / "ground_truth_experiment/calculation_runs"

if __name__ == "__main__":

    artn_out_files = glob.glob(str(experiment_dir / "run*/artn.out"), recursive=True)

    list_saddle_energy = []
    for artn_out_file_path in artn_out_files:

        try:
            with open(artn_out_file_path, "r") as fd:
                artn_output = fd.read()
                saddle_energy = get_saddle_energy(artn_output)
                list_saddle_energy.append(saddle_energy)
        except Exception:
            list_saddle_energy.append(np.nan)

    attempts_count = len(list_saddle_energy)
    failure_count = np.isnan(list_saddle_energy).sum()

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f"Ground Truth Saddle Point Energies for Amorphous Silicon\n"
                 f"Attempts: {attempts_count}, Failures: {failure_count}",)
    ax1 = fig.add_subplot(111)
    ax1.hist(list_saddle_energy, bins=100, color='green', alpha=0.5, histtype="stepfilled")

    ax1.set_xlabel("Saddle Point Energy (eV)")
    ax1.set_ylabel("Count")
    fig.tight_layout()

    plt.show()

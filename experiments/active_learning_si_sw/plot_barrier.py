from collections import defaultdict

import numpy as np
import yaml
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.artn.artn_outputs import \
    get_saddle_energy
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)

plt.style.use(PLOT_STYLE_PATH)

experiment = "noop"

top_dir = TOP_DIR / "experiments/active_learning_si_sw"
experiment_dir = TOP_DIR / "experiments/active_learning_si_sw" / experiment
reference_artn_output_file = top_dir / "Si-vac_sw_potential/artn.out"

list_run_ids = [1, 2, 3, 4, 5]
list_campaign_ids = [1, 2, 3, 4]

if __name__ == "__main__":
    with open(reference_artn_output_file, "r") as fd:
        artn_output = fd.read()
        E_gt = get_saddle_energy(artn_output)

    results = defaultdict(list)

    for run_id in list_run_ids:
        for campaign_id in list_campaign_ids:
            campaign_dir = experiment_dir / f"run{run_id}" / f"campaign_{campaign_id}"

            with open(campaign_dir / "campaign_details.yaml", "r") as fd:
                campaign_details = yaml.load(fd, Loader=yaml.FullLoader)

            final_round = campaign_details["final_round"]
            threshold = campaign_details["uncertainty_threshold"]

            final_round_dir = campaign_dir / f"round_{final_round}"
            artn_output_file = final_round_dir / "lammps_artn/artn.out"
            try:
                with open(artn_output_file, "r") as fd:
                    artn_output = fd.read()
                    E = get_saddle_energy(artn_output)
                    results[threshold].append(E)
            except Exception:
                print("Failed to Extract Saddle Energy")

    # error = E_gt - list_E[-1]

    list_thresholds = []
    list_mean = []
    list_std = []
    for threshold, list_energies in results.items():
        list_thresholds.append(threshold)
        list_mean.append(np.mean(list_energies))
        list_std.append(np.std(list_energies))

    list_mean = np.array(list_mean)
    list_std = np.array(list_std)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f"Saddle Point Energy From Active Learning\n Algorithm : {experiment}")
    ax1 = fig.add_subplot(111)

    ax1.semilogx(list_thresholds, list_mean, "ro-", ms=10, label="Mean Flare Value")

    for threshold, list_energies in results.items():
        n = len(list_energies)
        ax1.scatter(n * [threshold], list_energies, s=20, alpha=0.5, color="red")

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

    ax1.hlines(E_gt, *ax1.set_xlim(), linestyles="-", colors="green", label="SW value")

    ax1.legend(loc=3)
    fig.tight_layout()

    plt.show()

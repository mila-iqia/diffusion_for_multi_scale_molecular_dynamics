import pickle

import numpy as np
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)

plt.style.use(PLOT_STYLE_PATH)

experiment_dir = TOP_DIR / "experiments/active_learning/pretraining_flare/"
output_dir = experiment_dir / "validation_performance"

images_dir = experiment_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    with open(output_dir / "test_set_performance.pkl", "rb") as fd:
        results = pickle.load(fd)

    number_of_structures = results["number_of_structures"]
    sigma = results["sigma"]
    sigma_e = results["sigma_e"]
    sigma_f = results["sigma_f"]

    title_string = rf"{number_of_structures} structures, $\sigma$ = {sigma:2.1e}, \
    $\sigma_e$ = {sigma_e:2.1e}, $\sigma_f$ = {sigma_f:2.1e}"

    common_params = dict(linestyle="None", marker="o", markersize=5, mew=0, alpha=0.5)

    figsize = (1.5 * PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[1])
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"Evaluating Pretrained FLARE on Test Set\n{title_string}")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_xlabel("FLARE Uncertainty")
    ax1.set_ylabel("FLARE Force Error")
    ax2.set_xlabel("MAPPED FLARE Uncertainty")
    ax2.set_ylabel("MAPPED FLARE Force Error")

    xmax1, ymax1, xmax2, ymax2 = 0, 0, 0, 0

    # ================================================================================
    list_flare_force_errors = results["flare_all_force_errors"]
    list_flare_uncertainties = results["flare_all_uncertainties"]

    coeffs = np.polyfit(list_flare_uncertainties, list_flare_force_errors, deg=1)
    x = np.array([0.0, list_flare_uncertainties.max()])
    y = np.poly1d(coeffs)(x)

    ax1.hexbin(
        list_flare_uncertainties,
        list_flare_force_errors,
        gridsize=100,
        bins="log",
        cmap="inferno",
    )

    ax1.plot(x, y, "-", lw=2, label=rf"slope = {coeffs[0]:4.1f}")

    xmax1 = np.max([xmax1, list_flare_uncertainties.max()])
    ymax1 = np.max([ymax1, list_flare_force_errors.max()])

    # ================================================================================
    list_mapped_flare_force_errors = results["mapped_flare_all_force_errors"]
    list_mapped_flare_uncertainties = results["mapped_flare_all_uncertainties"]

    coeffs = np.polyfit(
        list_mapped_flare_uncertainties, list_mapped_flare_force_errors, deg=1
    )
    x = np.array([0.0, list_mapped_flare_uncertainties.max()])
    y = np.poly1d(coeffs)(x)

    ax2.hexbin(
        list_mapped_flare_uncertainties,
        list_mapped_flare_force_errors,
        gridsize=100,
        bins="log",
        cmap="inferno",
    )
    ax2.plot(x, y, "-", lw=2, label=rf"slope = {coeffs[0]:4.1f}")

    xmax2 = np.max([xmax2, list_mapped_flare_uncertainties.max()])
    ymax2 = np.max([ymax2, list_mapped_flare_force_errors.max()])

    ax1.legend(loc=0)
    ax1.set_xlim(xmin=0, xmax=xmax1)
    ax1.set_ylim(ymin=0, ymax=ymax1)

    ax2.legend(loc=0)
    ax2.set_xlim(xmin=0, xmax=xmax2)
    ax2.set_ylim(ymin=0, ymax=ymax2)

    fig.tight_layout()
    fig.savefig(images_dir / "errors_vs_uncertainties_on_test_set.png")

    # ================================================================================

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f"Comparing FLARE and MAPPED FLARE On Test Set\n{title_string}")
    ax = fig.add_subplot(111)
    ax.set_xlabel("FLARE Uncertainty")
    ax.set_ylabel("MAPPED FLARE Uncertainty")

    list_flare_uncertainties = results["flare_all_uncertainties"]
    list_mapped_flare_uncertainties = results["mapped_flare_all_uncertainties"]
    coeffs = np.polyfit(
        list_flare_uncertainties, list_mapped_flare_uncertainties, deg=1
    )
    x = np.array([0.0, list_flare_uncertainties.max()])
    y = np.poly1d(coeffs)(x)

    ax.hexbin(
        list_flare_uncertainties,
        list_mapped_flare_uncertainties,
        gridsize=100,
        cmap="inferno",
        bins="log",
    )

    ax.plot(x, y, "-", lw=2, label=rf"slope = {coeffs[0]:4.3f}")

    ax.legend(loc=0)
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)

    fig.tight_layout()
    fig.savefig(images_dir / "flare_vs_mapped_flare_uncertainties_on_test_set.png")

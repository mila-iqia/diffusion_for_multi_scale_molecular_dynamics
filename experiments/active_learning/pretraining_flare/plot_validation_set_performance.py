import numpy as np
import pandas as pd
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
    df = pd.read_pickle(output_dir / "validation_set_performance.pkl").sort_values(
        ["sigma", "number_of_structures"]
    )

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Training FLARE from scratch on Si 2x2x2.")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_title("Energy Errors")
    ax1.set_xlabel("Number of Training Structures")
    ax1.set_ylabel("Validation Energy RMSE (eV)")

    ax2.set_title("Force Errors")
    ax2.set_xlabel("Number of Training Structures")
    ax2.set_ylabel(r"Validation Mean Force RMSE (eV / $\AA$)")

    for sigma, group_df in df.groupby(by="sigma"):
        number_of_training_structures = group_df["number_of_structures"].values
        validation_energy_rmse = group_df["flare_energy_rmse"].values
        validation_mean_force_rmse = group_df["flare_mean_force_rmse"]

        mapped_validation_energy_rmse = group_df["mapped_flare_energy_rmse"].values
        mapped_validation_mean_force_rmse = group_df[
            "mapped_flare_mean_force_rmse"
        ].values

        (lines,) = ax1.semilogy(
            number_of_training_structures,
            validation_energy_rmse,
            "-",
            label=rf"$\sigma$ = {sigma}",
        )
        color = lines.get_color()

        ax1.semilogy(
            number_of_training_structures,
            mapped_validation_energy_rmse,
            "*",
            ms=10,
            color=color,
            label=rf"$\sigma$ = {sigma} (MAPPED)",
        )

        ax2.semilogy(
            number_of_training_structures, validation_mean_force_rmse, "-", color=color
        )

        ax2.semilogy(
            number_of_training_structures,
            mapped_validation_mean_force_rmse,
            "*",
            ms=10,
            color=color,
        )

    xmin = df["number_of_structures"].min() - 1
    xmax = df["number_of_structures"].max() + 1
    ax2.hlines(0.01, xmin, xmax, color="green", label="ARTn Force Threshold")

    for ax in [ax1, ax2]:
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin, xmax)

    handles, labels = ax1.get_legend_handles_labels()

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",  # Place the legend's center below the plots
        bbox_to_anchor=(
            0.5,
            -0.025,
        ),  # x=0.5 (center), y=-0.05 (just below the figure bottom)
        ncol=4,  # Number of columns for legend entries
        fancybox=True,
        shadow=True,
        borderaxespad=1.0,
    )

    plt.subplots_adjust(bottom=0.25)
    fig.savefig(images_dir / "errors_vs_number_of_structures.png")

    # ================================================================================

    sub_df = df[df["number_of_structures"] == 10]

    common_params = dict(linestyle="None", marker="o", markersize=5, mew=0, alpha=0.5)

    figsize = (1.5 * PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[1])
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        "Training FLARE from scratch on Si 2x2x2.\n Models trained with 10 structures."
    )
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_xlabel("FLARE Uncertainty")
    ax1.set_ylabel("FLARE Force Error")
    ax2.set_xlabel("MAPPED FLARE Uncertainty")
    ax2.set_ylabel("MAPPED FLARE Force Error")

    xmax1, ymax1, xmax2, ymax2 = 0, 0, 0, 0

    for sigma, sigma_df in sub_df.groupby(by="sigma"):

        # ================================================================================
        list_flare_force_errors = sigma_df["flare_all_force_errors"].values[0]
        list_flare_uncertainties = sigma_df["flare_all_uncertainties"].values[0]

        coeffs = np.polyfit(list_flare_uncertainties, list_flare_force_errors, deg=1)
        x = np.array([0.0, list_flare_uncertainties.max()])
        y = np.poly1d(coeffs)(x)

        (line,) = ax1.plot(
            list_flare_uncertainties,
            list_flare_force_errors,
            **common_params,
            label=rf"$\sigma$ = {sigma}, slope = {coeffs[0]:4.1f}",
        )

        ax1.plot(x, y, "-", c=line.get_color(), lw=4, label="__nolabel__")
        xmax1 = np.max([xmax1, list_flare_uncertainties.max()])
        ymax1 = np.max([ymax1, list_flare_force_errors.max()])

        # ================================================================================
        list_mapped_flare_force_errors = sigma_df[
            "mapped_flare_all_force_errors"
        ].values[0]
        list_mapped_flare_uncertainties = sigma_df[
            "mapped_flare_all_uncertainties"
        ].values[0]

        coeffs = np.polyfit(
            list_mapped_flare_uncertainties, list_mapped_flare_force_errors, deg=1
        )
        x = np.array([0.0, list_mapped_flare_uncertainties.max()])
        y = np.poly1d(coeffs)(x)
        (line,) = ax2.plot(
            list_mapped_flare_uncertainties,
            list_mapped_flare_force_errors,
            **common_params,
            label=rf"$\sigma$ = {sigma}, slope = {coeffs[0]:4.1f}",
        )
        ax2.plot(x, y, "-", c=line.get_color(), lw=4, label="__nolabel__")

        xmax2 = np.max([xmax2, list_mapped_flare_uncertainties.max()])
        ymax2 = np.max([ymax2, list_mapped_flare_force_errors.max()])

    ax1.legend(loc=0)
    ax1.set_xlim(xmin=0, xmax=xmax1)
    ax1.set_ylim(ymin=0, ymax=ymax1)

    ax2.legend(loc=0)
    ax2.set_xlim(xmin=0, xmax=xmax2)
    ax2.set_ylim(ymin=0, ymax=ymax2)

    fig.tight_layout()
    fig.savefig(images_dir / "errors_vs_uncertainties.png")

    # ================================================================================

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Comparing FLARE and MAPPED FLARE")
    ax = fig.add_subplot(111)
    ax.set_xlabel("FLARE Uncertainty")
    ax.set_ylabel("MAPPED FLARE Uncertainty")

    for sigma, sigma_df in sub_df.groupby(by="sigma"):

        list_flare_uncertainties = sigma_df["flare_all_uncertainties"].values[0]
        list_mapped_flare_uncertainties = sigma_df[
            "mapped_flare_all_uncertainties"
        ].values[0]
        coeffs = np.polyfit(
            list_flare_uncertainties, list_mapped_flare_uncertainties, deg=1
        )
        x = np.array([0.0, list_flare_uncertainties.max()])
        y = np.poly1d(coeffs)(x)

        (line,) = ax.plot(
            list_flare_uncertainties,
            list_mapped_flare_uncertainties,
            "o",
            label=rf"$\sigma$ = {sigma}, slope = {coeffs[0]:4.3f}",
        )

        ax.plot(x, y, "-", c=line.get_color(), lw=4, label="__nolabel__")

    ax.legend(loc=0)
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)

    fig.tight_layout()
    fig.savefig(images_dir / "flare_vs_mapped_flare_uncertainties.png")

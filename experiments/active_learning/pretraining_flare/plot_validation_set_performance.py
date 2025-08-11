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
    df = pd.read_pickle(output_dir / "validation_set_performance.pkl")

    for idx, (sigma_e, group_df) in enumerate(df.groupby(by="sigma_e")):

        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        fig.suptitle(rf"Training FLARE from scratch on Si 2x2x2: $\sigma_e$ = {sigma_e:3.2e}")
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.set_title("Energy Errors")
        ax1.set_xlabel("Number of Training Structures")
        ax1.set_ylabel("Validation Energy RMSE (eV)")

        ax2.set_title("Force Errors")
        ax2.set_xlabel("Number of Training Structures")
        ax2.set_ylabel(r"Validation Mean Force RMSE (eV / $\AA$)")

        for sigma_f, sub_df in group_df.groupby(by="sigma_f"):
            sub_df = sub_df.sort_values(by='number_of_structures')
            number_of_training_structures = sub_df["number_of_structures"].values
            validation_energy_rmse = sub_df["energy_rmse"].values
            validation_mean_force_rmse = sub_df["mean_force_rmse"]

            (lines,) = ax1.semilogy(
                number_of_training_structures,
                validation_energy_rmse,
                "-",
                label=rf"$\sigma_f$ = {sigma_f:3.1e}",
            )
            color = lines.get_color()

            ax2.semilogy(
                number_of_training_structures, validation_mean_force_rmse, "-", color=color
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
            ),
            ncol=5,  # Number of columns for legend entries
            fancybox=True,
            shadow=True,
            borderaxespad=1.0,
            fontsize=10
        )

        plt.subplots_adjust(bottom=0.25)
        fig.savefig(images_dir / f"errors_vs_number_of_structures_{idx}.png")

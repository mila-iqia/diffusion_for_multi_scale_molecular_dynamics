import pickle

import matplotlib.pyplot as plt
import numpy as np

from diffusion_for_multi_scale_molecular_dynamics.analysis import \
    PLOT_STYLE_PATH
from experiments.analysis.analytic_score.exploring_langevin_generator import \
    LANGEVIN_EXPLORATION_DIRECTORY

plt.style.use(PLOT_STYLE_PATH)

supercell_factor = 2
pickle_directory = (
    LANGEVIN_EXPLORATION_DIRECTORY
    / f"Si_{supercell_factor}x{supercell_factor}x{supercell_factor}"
)

title = f"Si {supercell_factor}x{supercell_factor}x{supercell_factor}"

sigma_d = 0.001
sigma_min = 0.001
if __name__ == "__main__":
    list_q = np.linspace(0, 1, 101)

    with open(pickle_directory / f"exact_energies_Sd={sigma_d}.pkl", "rb") as fd:
        exact_results = pickle.load(fd)

    FIG_SIZE = (7.2, 7.2)
    fig = plt.figure(figsize=FIG_SIZE)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    fig.suptitle(f"{title} : $\\sigma_d$ = {sigma_d}, $\\sigma_m$ = {sigma_min}")

    list_energy_quantiles = np.quantile(exact_results["energies"], list_q)
    for ax in [ax1, ax2, ax3]:
        ax.plot(
            100 * list_q,
            list_energy_quantiles,
            "-",
            lw=10,
            color="green",
            label="Exact",
        )

    min = list_energy_quantiles[0]
    max = list_energy_quantiles[-1]
    range = max - min

    min = min - 1
    max = max + 10000

    for number_of_corrector_steps, ax in zip([1, 10, 100], [ax1, ax2, ax3]):
        ax.set_title(f"Corrector steps: {number_of_corrector_steps}")

        lw = 8
        for total_time_steps, color in zip([10, 100, 1000], ["r", "y", "b"]):
            name = (
                f"sampled_energies_Sd={sigma_d}_Sm={sigma_min}"
                f"_S={total_time_steps}_C={number_of_corrector_steps}.pkl"
            )
            filepath = pickle_directory / name
            if not filepath.is_file():
                print(f"File {name} is missing. Moving on.")
                continue
            with open(filepath, "rb") as fd:
                sampled_results = pickle.load(fd)
            energies = sampled_results.pop("energies")
            list_energy_quantiles = np.quantile(energies, list_q)
            ax.plot(
                100 * list_q,
                list_energy_quantiles,
                "-.",
                lw=lw,
                c=color,
                label=f"total time steps = {total_time_steps}",
            )
            lw = lw / 1.5

            ax.set_xlabel("Quantile (%)")
            ax.set_ylabel("Energy Quantile (eV)")

            ax.set_xlim([-0.01, 100.01])
            ax.set_ylim([min, max])
            ax.legend(loc=0)
        fig.tight_layout()
    plt.show()

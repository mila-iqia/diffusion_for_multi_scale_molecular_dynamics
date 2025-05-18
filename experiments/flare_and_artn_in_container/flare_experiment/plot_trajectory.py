from pathlib import Path

import numpy as np
import yaml
from flare_experiment.utilities import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from flare_experiment.utilities.utils import parse_thermo_fields
from matplotlib import pyplot as plt

plt.style.use(PLOT_STYLE_PATH)

artn_experiment_path = Path("/home/user/experiments/flare_experiment/artn/")

trajectory_index = 3
trajectory_path = artn_experiment_path / f"trajectory{trajectory_index}"

dump_file_path = str(trajectory_path / "dump.yaml")

if __name__ == "__main__":

    list_energies = []
    list_unc = []
    with open(dump_file_path, "r") as f:
        data = yaml.load_all(f, Loader=yaml.CLoader)
        for doc in data:
            thermo_dict = parse_thermo_fields(doc)
            potential_energy = thermo_dict['PotEng']
            unc = thermo_dict['v_max_unc']
            list_energies.append(potential_energy)
            list_unc.append(unc)

    list_energies = np.array(list_energies)
    n_steps = len(list_energies)
    e_min = np.min(list_energies)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f"FLARE Energy along ARTn trajectory {trajectory_index} for vac-Si Example")
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(list_energies, 'bo', label='ARTn steps')
    ax1.set_ylabel("Energy")
    ax1.hlines(e_min, *ax1.get_xlim(), colors='k', linestyles='--', label="Minimum Energy")

    ax2.semilogy(list_unc, 'bo')
    ax2.set_ylabel("Max Uncertainty")

    for ax in [ax1, ax2]:
        ax.set_xlabel("ARTn step")
        ax.set_xlim(0, n_steps)
        ax.legend(loc=0)
    fig.tight_layout()
    plt.show()
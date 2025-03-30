"""Silicon phonon Density of States.

The displacement covariance is related to the phonon dynamical matrix.
Here we extract the corresponding phonon density of state, based on this covariance,
to see if the energy scales match up.
"""

import matplotlib.pyplot as plt
import torch

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from experiments.dataset_analysis import RESULTS_DIR

plt.style.use(PLOT_STYLE_PATH)

# Define some constants.
kelvin_in_Ha = 0.000_003_166_78
T_in_kelvin = 300.0
bohr_in_angst = 0.529177
Si_mass = 28.0855
proton_mass = 1836.152673426

Ha_in_meV = 27211.0

THz_in_meV = 4.136

acell = 5.43


dataset_name_3x3x3 = "Si_diffusion_3x3x3"
dataset_name_2x2x2 = "Si_diffusion_2x2x2"
dataset_name_1x1x1 = "Si_diffusion_1x1x1"

list_dataset_names = [dataset_name_1x1x1, dataset_name_2x2x2, dataset_name_3x3x3]
list_scale_factor = [1, 2, 3]

list_colors = ['red', 'green', 'blue']

covariance_dir = RESULTS_DIR / "covariances"

if __name__ == "__main__":
    kBT = kelvin_in_Ha * T_in_kelvin
    a = acell / bohr_in_angst

    M = Si_mass * proton_mass

    list_omegas = []

    for scale_factor, dataset_name in zip(list_scale_factor, list_dataset_names):
        constant = M * (scale_factor * a) ** 2 / kBT / Ha_in_meV ** 2
        covariance_file = covariance_dir / f"covariance_{dataset_name}.pkl"
        sigma = torch.load(covariance_file)
        sigma_inv = torch.linalg.pinv(sigma)
        Omega = sigma_inv / constant
        omega2 = torch.linalg.eigvalsh(Omega)
        omega_in_meV = torch.sqrt(torch.abs(omega2))
        list_omegas.append(omega_in_meV)

    max_hw = torch.max(torch.cat(list_omegas).max()) / THz_in_meV

    bins = torch.linspace(0.0, max_hw, 100)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Eigenvalues of Dynamical Matrix, from Displacement Covariance")
    ax = fig.add_subplot(111)

    ax.set_xlim(0, max_hw + 1)
    ax.set_xlabel(r"$\hbar \omega$ (THz)")
    ax.set_ylabel("Density")

    for omega_in_meV, dataset_name, color in zip(list_omegas, list_dataset_names, list_colors):
        ax.hist(
            omega_in_meV / THz_in_meV,
            bins=bins,
            label=dataset_name,
            color=color,
            alpha=0.25,
            histtype="stepfilled",
            density=True
        )

    ax.legend(loc=0)
    fig.savefig(RESULTS_DIR / "phonons_DOS.png")
    plt.show()

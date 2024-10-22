"""Silicon phonon Density of States.

The displacement covariance is related to the phonon dynamical matrix.
Here we extract the corresponding phonon density of state, based on this covariance,
to see if the energy scales match up.
"""

import matplotlib.pyplot as plt
import torch

from diffusion_for_multi_scale_molecular_dynamics import ANALYSIS_RESULTS_DIR
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)

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


dataset_name_2x2x2 = "si_diffusion_2x2x2"
dataset_name_1x1x1 = "si_diffusion_1x1x1"

output_dir = ANALYSIS_RESULTS_DIR / "covariances"
output_dir.mkdir(exist_ok=True)


if __name__ == "__main__":
    kBT = kelvin_in_Ha * T_in_kelvin
    a = acell / bohr_in_angst

    M = Si_mass * proton_mass

    constant_1x1x1 = M * a**2 / kBT / Ha_in_meV**2
    constant_2x2x2 = M * (2.0 * a) ** 2 / kBT / Ha_in_meV**2

    covariance_file_1x1x1 = output_dir / f"covariance_{dataset_name_1x1x1}.pkl"
    sigma_1x1x1 = torch.load(covariance_file_1x1x1)
    sigma_inv_1x1x1 = torch.linalg.pinv(sigma_1x1x1)
    Omega_1x1x1 = sigma_inv_1x1x1 / constant_1x1x1
    omega2_1x1x1 = torch.linalg.eigvalsh(Omega_1x1x1)
    list_omega_in_meV_1x1x1 = torch.sqrt(torch.abs(omega2_1x1x1))

    covariance_file_2x2x2 = output_dir / f"covariance_{dataset_name_2x2x2}.pkl"
    sigma_2x2x2 = torch.load(covariance_file_2x2x2)
    sigma_inv_2x2x2 = torch.linalg.pinv(sigma_2x2x2)
    Omega_2x2x2 = sigma_inv_2x2x2 / constant_2x2x2
    omega2_2x2x2 = torch.linalg.eigvalsh(Omega_2x2x2)
    list_omega_in_meV_2x2x2 = torch.sqrt(torch.abs(omega2_2x2x2))

    max_hw = torch.max(list_omega_in_meV_2x2x2) / THz_in_meV

    bins = torch.linspace(0.0, max_hw, 50)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Eigenvalues of Dynamical Matrix, from Displacement Covariance")
    ax = fig.add_subplot(111)

    ax.set_xlim(0, max_hw + 1)
    ax.set_xlabel(r"$\hbar \omega$ (THz)")
    ax.set_ylabel("Count")

    ax.hist(
        list_omega_in_meV_1x1x1 / THz_in_meV,
        bins=bins,
        label="Si 1x1x1",
        color="green",
        alpha=0.5,
    )
    ax.hist(
        list_omega_in_meV_2x2x2 / THz_in_meV,
        bins=bins,
        label="Si 2x2x2",
        color="blue",
        alpha=0.25,
    )
    ax.legend(loc=0)
    plt.show()

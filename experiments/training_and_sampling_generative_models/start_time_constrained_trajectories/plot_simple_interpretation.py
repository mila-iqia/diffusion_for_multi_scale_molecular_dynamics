from pathlib import Path

import numpy as np
import scipy.optimize as so
import scipy.special as ss
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)

plt.style.use(PLOT_STYLE_PATH)


def get_wigner_seitz_radius(volume: float, number_of_atoms: int):
    """Get WS radius."""
    r0 = (3.0 / 4.0 / np.pi * volume / number_of_atoms) ** (1.0 / 3.0)
    return r0


def get_probability_to_sample_in_wigner_seitz_sphere(
    sigma: float, wigner_seitz_radius: float
):
    """Get probability to sample in wigner seitz sphere."""
    xi = wigner_seitz_radius / sigma
    probability = ss.erf(xi / np.sqrt(2.0)) - np.sqrt(2.0 / np.pi) * xi * np.exp(
        -(xi**2) / 2.0
    )
    return probability


def get_double_assigment_probability(sigma: float, volume: float, number_of_atoms: int):
    """Get double assigment probability."""
    r0 = get_wigner_seitz_radius(volume, number_of_atoms)
    p0 = get_probability_to_sample_in_wigner_seitz_sphere(sigma, r0)

    q = 1.0 - p0**number_of_atoms
    return q


image_directory = Path(__file__).parent / "images"
image_directory.mkdir(exist_ok=True)

number_of_atoms = 216
volume = 1.0
sigma_min = 0.0001
sigma_max = 0.20

if __name__ == "__main__":

    list_sigma = np.linspace(sigma_min, sigma_max, 1001)
    list_q = [
        get_double_assigment_probability(sigma, volume, number_of_atoms)
        for sigma in list_sigma
    ]

    function_to_minimize = (
        lambda sigma: get_double_assigment_probability(sigma, volume, number_of_atoms)
        - 0.5
    )
    sigma0 = 0.025
    sol = so.root(function_to_minimize, x0=sigma0)
    sigma_mid_point = sol.x[0]

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Probability of Randomly Assigning Multiple Atoms to Same Cell")
    ax = fig.add_subplot(111)

    ax.plot(list_sigma, list_q, "-")

    label = rf"50% Probability at $\sigma$ = {sigma_mid_point: 3.4f}"
    ax.vlines(sigma_mid_point, 0, 1, color="g", linestyle="--", label=label)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"Q")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim(0, 0.06)
    ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()

    output_file = image_directory / "short_edges_simple_interpretation.png"
    fig.savefig(output_file)
    plt.close(fig)

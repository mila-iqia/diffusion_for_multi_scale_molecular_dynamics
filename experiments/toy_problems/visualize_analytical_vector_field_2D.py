import einops
import numpy as np
import torch
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import \
    VarianceScheduler
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.score.wrapped_gaussian_score import \
    get_log_wrapped_gaussians
from experiments.toy_problems import RESULTS_DIR
from experiments.toy_problems.visualization_utils import \
    generate_vector_field_video

plt.style.use(PLOT_STYLE_PATH)

output_file_path = RESULTS_DIR / "analytical_score.mp4"

sigma_min = 0.001
sigma_max = 0.2
sigma_d = 0.01


spatial_dimension = 1
x0_1 = 0.25
x0_2 = 0.75

if __name__ == "__main__":
    noise_parameters = NoiseParameters(
        total_time_steps=100, sigma_min=sigma_min, sigma_max=sigma_max
    )

    score_network_parameters = AnalyticalScoreNetworkParameters(
        number_of_atoms=2,
        num_atom_types=1,
        kmax=5,
        equilibrium_relative_coordinates=[[x0_1], [x0_2]],
        sigma_d=sigma_d,
        spatial_dimension=spatial_dimension,
        use_permutation_invariance=True,
    )

    analytical_score_network = AnalyticalScoreNetwork(score_network_parameters)

    generate_vector_field_video(
        axl_network=analytical_score_network,
        analytical_score_network=analytical_score_network,
        noise_parameters=noise_parameters,
        output_file_path=output_file_path,
    )

    # Plot the marginal probability distribution.
    x = torch.linspace(0, 1, 1001)
    relative_coordinates1 = einops.rearrange(x - x0_1, "batch -> batch 1 1")
    relative_coordinates2 = einops.rearrange(x - x0_2, "batch -> batch 1 1")

    list_times = torch.linspace(0.0, 1.0, 5)
    list_sigmas = VarianceScheduler(noise_parameters).get_sigma(list_times).numpy()
    list_sigmas[0] = 0.0

    list_marginal_probabilities = []
    for time, sigma in zip(list_times, list_sigmas):

        effective_sigmas = np.sqrt(sigma_d**2 + sigma**2) * torch.ones_like(
            relative_coordinates1
        )

        log_wrapped_gaussian1 = get_log_wrapped_gaussians(
            relative_coordinates1, effective_sigmas, kmax=5
        )
        wrapped_gaussian1 = torch.exp(log_wrapped_gaussian1)
        log_wrapped_gaussian2 = get_log_wrapped_gaussians(
            relative_coordinates2, effective_sigmas, kmax=5
        )
        wrapped_gaussian2 = torch.exp(log_wrapped_gaussian2)
        marginal_probability = 0.5 * (wrapped_gaussian1 + wrapped_gaussian2)

        list_marginal_probabilities.append(marginal_probability)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(r"Marginal Probability Distribution: $\sigma_{d}$ = " + f"{sigma_d}")
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    lw = 4
    for t, sigma, marginal_probability in zip(
        list_times, list_sigmas, list_marginal_probabilities
    ):
        label = "$t$ = " + f"{t}" + r", $\sigma(t)$ =" + f"{sigma:3.2f}"
        ax.plot(x, marginal_probability, lw=lw, label=label)
        lw = 2

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(ymin=-0.01)
    ax.legend(loc=0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$P(x)$")
    plt.savefig(RESULTS_DIR / "marginal_probability_distribution.png")
    plt.close(fig)

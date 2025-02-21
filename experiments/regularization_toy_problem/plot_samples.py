import torch
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.namespace import \
    AXL_COMPOSITION
from experiments.regularization_toy_problem import RESULTS_DIR
from experiments.regularization_toy_problem.visualization_utils import \
    plot_2d_samples

if __name__ == "__main__":

    samples = torch.load(
        RESULTS_DIR / "analytical_score_network_samples" / "samples.pt"
    )
    relative_coordinates = samples[AXL_COMPOSITION].X.squeeze(-1)

    fig = plot_2d_samples(relative_coordinates)
    fig.suptitle("Samples drawn with Analytical Score Network")
    fig.savefig(RESULTS_DIR / "analytical_score_samples.png")
    plt.close(fig)

    samples = torch.load(
        RESULTS_DIR / "mlp_score_network_samples_no_regularizer" / "samples.pt"
    )
    relative_coordinates = samples[AXL_COMPOSITION].X.squeeze(-1)

    fig = plot_2d_samples(relative_coordinates)
    fig.suptitle("Samples drawn with MLP Score Network")
    fig.savefig(RESULTS_DIR / "mlp_score_samples_no_regularizer.png")
    plt.close(fig)

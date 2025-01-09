import glob
from pathlib import Path

from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.analysis import \
    PLOT_STYLE_PATH
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.sample_diffusion import \
    get_axl_network
from experiments.regularization_toy_problem import EXPERIMENTS_DIR, RESULTS_DIR
from experiments.regularization_toy_problem.visualization_utils import \
    generate_vector_field_video

plt.style.use(PLOT_STYLE_PATH)


output_file_path = RESULTS_DIR / "mlp_score_no_regularizer.mp4"
checkpoint_path = glob.glob(str(EXPERIMENTS_DIR / "no_regularizer/**/last_model*.ckpt"), recursive=True)[0]

sigma_min = 0.001
sigma_max = 0.2
sigma_d = 0.01

spatial_dimension = 1
x0_1 = 0.25
x0_2 = 0.75

if __name__ == "__main__":
    checkpoint_name = Path(checkpoint_path).name
    axl_network = get_axl_network(checkpoint_path)

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
        axl_network=axl_network,
        analytical_score_network=analytical_score_network,
        noise_parameters=noise_parameters,
        output_file_path=output_file_path,
    )

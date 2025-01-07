import subprocess
import tempfile
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.analysis import \
    PLOT_STYLE_PATH
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import \
    AnalyticalScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import \
    VarianceScheduler
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from experiments.two_atoms_in_one_dimension.utils import \
    get_2d_vector_field_figure

plt.style.use(PLOT_STYLE_PATH)


def generate_vector_field_video(
    axl_network: ScoreNetwork,
    analytical_score_network: AnalyticalScoreNetwork,
    noise_parameters: NoiseParameters,
    output_file_path: Path,
):
    """This method will generate a a video of 2D vector fields."""
    list_times = torch.linspace(0.0, 1.0, noise_parameters.total_time_steps)
    list_sigmas = VarianceScheduler(noise_parameters).get_sigma(list_times).numpy()

    spatial_dimension = analytical_score_network.spatial_dimension
    sigma_d = np.sqrt(analytical_score_network.sigma_d_square)

    number_of_spatial_points = 100  # let's be opinionated about this.
    n1 = number_of_spatial_points
    n2 = number_of_spatial_points

    x1 = torch.linspace(0, 1, n1)
    x2 = torch.linspace(0, 1, n2)

    X1, X2_ = torch.meshgrid(x1, x2, indexing="xy")
    X2 = torch.flip(X2_, dims=[0])

    relative_coordinates = einops.repeat(
        [X1, X2], "natoms n1 n2 -> (n1 n2) natoms space", space=spatial_dimension
    ).contiguous()
    relative_coordinates = map_relative_coordinates_to_unit_cell(relative_coordinates)

    forces = torch.zeros_like(relative_coordinates)
    batch_size, natoms, _ = relative_coordinates.shape

    atom_types = torch.ones(batch_size, natoms, dtype=torch.int64)

    list_ground_truth_probabilities = []
    list_sigma_normalized_scores = []
    for time, sigma in tqdm(zip(list_times, list_sigmas), "SIGMAS"):
        grid_sigmas = sigma * torch.ones_like(relative_coordinates)
        flat_probabilities, flat_normalized_scores = (
            analytical_score_network.get_probabilities_and_normalized_scores(
                relative_coordinates, grid_sigmas
            )
        )
        probabilities = einops.rearrange(
            flat_probabilities, "(n1 n2) -> n1 n2", n1=n1, n2=n2
        )
        list_ground_truth_probabilities.append(probabilities)

        sigma_t = sigma * torch.ones(batch_size, 1)
        times = time * torch.ones(batch_size, 1)
        unit_cell = torch.ones(batch_size, 1, 1)

        composition = AXL(
            A=atom_types,
            X=relative_coordinates,
            L=torch.zeros_like(relative_coordinates),
        )

        batch = {
            NOISY_AXL_COMPOSITION: composition,
            NOISE: sigma_t,
            TIME: times,
            UNIT_CELL: unit_cell,
            CARTESIAN_FORCES: forces,
        }

        model_predictions = axl_network(batch)
        sigma_normalized_scores = einops.rearrange(
            model_predictions.X.detach(),
            "(n1 n2) natoms space -> n1 n2 natoms space",
            n1=n1,
            n2=n2,
        )

        list_sigma_normalized_scores.append(sigma_normalized_scores)

    sigma_normalized_scores = torch.stack(list_sigma_normalized_scores).squeeze(-1)
    ground_truth_probabilities = torch.stack(list_ground_truth_probabilities)

    # ================================================================================

    s = 2
    with tempfile.TemporaryDirectory() as tmpdirname:

        tmp_dir = Path(tmpdirname)

        for time_idx in tqdm(range(len(list_times)), "VIDEO"):
            sigma_t = list_sigmas[time_idx]
            time = list_times[time_idx].item()

            fig = get_2d_vector_field_figure(
                X1=X1,
                X2=X2,
                probabilities=ground_truth_probabilities[time_idx],
                sigma_normalized_scores=sigma_normalized_scores[time_idx],
                time=time,
                sigma_t=sigma_t,
                sigma_d=sigma_d,
                supsampling_scale=s,
            )

            output_image = tmp_dir / f"vector_field_{time_idx}.png"
            fig.savefig(output_image)
            plt.close(fig)

        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # ffmpeg -r 10  -start_number 0 -i vector_field_%d.png -vcodec libx264 -pix_fmt yuv420p mlp_vector_field.mp4
        commands = [
            "ffmpeg",
            "-r",
            "10",
            "-start_number",
            "0",
            "-i",
            str(tmp_dir / "vector_field_%d.png"),
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_file_path),
        ]

        # This subprocess should create the video.
        subprocess.run(commands, capture_output=True, text=True)

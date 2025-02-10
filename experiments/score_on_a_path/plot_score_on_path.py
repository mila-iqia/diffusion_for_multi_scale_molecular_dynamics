import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops import einsum

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL

plt.style.use(PLOT_STYLE_PATH)

ROOT_DIR = Path(
    "/Users/simonblackburn/projects/courtois2024/experiments/score_on_a_path"
)

data_path = ROOT_DIR / "model_predictions.pt"
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

UNIT_CELL_SIZE = 10.86  # cell size in angstrom


def main():
    """Create plots and video of scores as a function of distance & sigma."""
    data = torch.load(data_path, map_location="cpu")

    sigmas = data["sigma"]  # tensor of shape [num_time_steps]

    # trajectories:  # list of AXLs of len 101 - each is a different iteration of the movement on the path
    # each AXL.X is of shape [num_atoms, spatial_dimension] - no batch size
    trajectories = data["trajectories"]

    # the model_predictions are also stored in a list of len num_space_steps
    # each AXL.X is of shape [num_time_step, num_atoms, spatial_dimension]
    model_predictions = data["model_predictions"]  # list of AXLS of len num_space_steps

    # jacobians are stored in a single tensor of shape
    # [num_space_steps, num_time_steps, num_atoms * spatial_dimension, num_atoms * spatial_dimension]
    # jacobians = data["jacobians"]
    # we will need to take the diagonal elements with a sum over spatial dimension to get the divergence on each atom

    # get the movement direction
    unnormalized_direction = (
        data["trajectories"][0].X - data["trajectories"][-1].X
    ).sum(dim=0)
    displacement_length = torch.linalg.norm(unnormalized_direction)
    unit_displacement = unnormalized_direction / displacement_length
    d_ang = displacement_length.item() * UNIT_CELL_SIZE  # displacement in Angstrom

    moved_atom_idx = (
        ((data["trajectories"][0].X - data["trajectories"][-1].X) ** 2)
        .sum(dim=1)
        .argmax()
        .item()
    )

    projected_scores = []
    for trajectory_axl, score_axl in zip(trajectories, model_predictions):
        projected_score = get_projected_score(
            score_axl, unit_vector=unit_displacement, target_idx=moved_atom_idx
        )
        projected_scores.append(projected_score)
    projected_scores = torch.stack(
        projected_scores
    )  # [num_space_steps, num_time_steps]

    projected_scores_plot_dir = FIGURES_DIR / "projected_scores_target_atom"
    projected_scores_plot_dir.mkdir(parents=True, exist_ok=True)
    for i in range(projected_scores.shape[1]):
        plot_name = projected_scores_plot_dir / f"score_{i}.png"
        plot_projected_scores(projected_scores, sigmas, i, d_ang, output=plot_name)
    video_name = projected_scores_plot_dir / "projected_scores_target_atom.mp4"
    make_video(projected_scores_plot_dir, video_name)


def get_projected_score(
    score_axl: AXL, unit_vector: torch.Tensor, target_idx: int
) -> torch.Tensor:
    """Find the score of the target atom and select those of the target atom."""
    score_x = score_axl.X[
        :, target_idx, :
    ]  # num_time_steps, num_atoms, spatial_dimension
    projected_score = einsum(score_x, unit_vector, "n s, s -> n")
    return projected_score  # num_time_steps


def plot_projected_scores(
    projected_scores: torch.Tensor,
    sigmas: torch.Tensor,
    time_idx: int,
    d_length: float,
    output: Path,
):
    """Plot the score along the path."""
    n_space_steps, n_time_steps = projected_scores.shape

    # need to tweak the figure size to make a video - ffmpeg needs an even number of pixels for the height
    fig = plt.figure(figsize=(2.0 * PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[1] + 0.01))

    fig.suptitle(r"$\sigma$ normalized score projected on atomic collapse path")

    current_sigma = sigmas[time_idx].item()

    space_steps = [
        (n_space_steps - x) * d_length / (n_space_steps - 1)
        for x in range(n_space_steps)
    ]

    ax = fig.add_subplot(122)
    im = ax.contourf(space_steps, sigmas, projected_scores.T)

    ax.set_xlabel(r"Distance between atoms $\AA$")
    ax.set_ylabel(r"$\sigma$")

    ax.set_yscale("log")
    ax.axhline(current_sigma, ls=":")
    plt.colorbar(im, label="sigma-normalized score")

    ax1 = fig.add_subplot(121)
    ax1.plot(space_steps, projected_scores[:, time_idx])
    ax1.set_xlabel(r"Distance between atoms ($\AA$)")
    ax1.set_ylabel(r"$\sigma$-normalized score")
    ax1.set_title(r"$\sigma$ =" + f"{current_sigma:0.4f}")
    ax1.axhline(y=0.0, ls="dashed")
    ax1.set_ylim(projected_scores.min() - 0.01, projected_scores.max() + 0.01)
    ax1.set_xlim(0, space_steps[0])
    plt.savefig(output, dpi=100)
    plt.close()


def make_video(plot_path, output_file_path):
    """Create a video from figures in a folder."""
    commands = [
        "ffmpeg",
        "-r",
        "10",
        "-start_number",
        "0",
        "-i",
        str(plot_path / "score_%d.png"),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_file_path),
    ]
    # This subprocess should create the video.
    subprocess.run(commands, capture_output=True, text=True)


if __name__ == "__main__":
    main()

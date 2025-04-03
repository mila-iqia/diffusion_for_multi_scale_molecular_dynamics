"""Utilities.

This module contains helper functions for the notebook tutorials.
"""
import subprocess
from pathlib import Path
from typing import List

import einops
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.analysis import \
    PLEASANT_FIG_SIZE


def get_2d_grid_equilibrium_relative_coordinates(n: int) -> List[List[float]]:
    """Get regular 2D grid of points."""
    linear_positions = torch.arange(n) / n + 0.5 / n
    u1, u2 = torch.meshgrid(linear_positions, linear_positions, indexing="ij")
    equilibrium_relative_coordinates = einops.rearrange([u1, u2], "d n1 n2 -> (n1 n2) d")
    return list(list(x) for x in equilibrium_relative_coordinates.numpy())


def compute_total_distance(relative_coordinates: torch.Tensor, reference_relative_coordinates: torch.Tensor) -> float:
    """Compute total distance.

    This method computes the "total distance" between two configurations, accounting for periodicity,
    by comparing coordinates in order.

    Args:
        relative_coordinates: the relative coordinates of a configuration.
            Dimension [number_of_atoms, spatial_dimension]
        reference_relative_coordinates: the reference relative coordinates of a configuration.
            Dimension [number_of_atoms, spatial_dimension]

    Returns:
        Total distance: the total distance between the relative coordinates and the reference, in reduced units.
    """
    raw_displacements = relative_coordinates - reference_relative_coordinates
    augmented_displacements = [raw_displacements - 1.0, raw_displacements, raw_displacements + 1.0]

    squared_displacements = einops.rearrange(augmented_displacements, "c n d -> (n d) c")**2

    total_displacement = torch.sqrt(squared_displacements.min(dim=1).values.sum())
    return total_displacement.item()


def plot_2d_relative_coordinates(relative_coordinates: torch.Tensor,
                                 reference_relative_coordinates: torch.Tensor,
                                 constrained_relative_coordinates: torch.Tensor,
                                 sigma_d: float) -> plt.Figure:
    """Plot 2D relative coordinates.

    Create a matplotlib figure showing the relative coordinates and the reference relative coordinates.
    It is assumed that the spatial dimension is 2, so that it is possible to create a 2D plot of coordinates.

    It is assumed that the reference dataset is distributed according to an isotropic Gaussian of width sigma_d
    centered on the reference coordinates.

    Args:
        relative_coordinates: relative coordinates of a configuration. Dimension [number_of_atoms, spatial_dimension]
        reference_relative_coordinates: center of reference Gaussian  distribution.
            Dimension [number_of_atoms, spatial_dimension]
        constrained_relative_coordinates: constrained relative coordinates for repaint. Dimensions
            [number of targets, spatial_dimension]
        sigma_d: the standard deviation of the Gaussian distribution.

    Returns:
        figure: the matplotlib figure showing the relative coordinates and the reference relative coordinates.
    """
    figsize = (1.4 * PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[0])
    fig1 = plt.figure(figsize=figsize)

    ax1 = fig1.add_subplot(111, aspect="equal")

    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)

    ax1.scatter(relative_coordinates[:, 0],
                relative_coordinates[:, 1],
                c='red',
                linewidths=0,
                label='$x(t)$')

    ax1.scatter(reference_relative_coordinates[:, 0],
                reference_relative_coordinates[:, 1],
                c='blue',
                linewidths=0,
                alpha=0.5,
                label=r'$\mu$')

    if len(constrained_relative_coordinates) > 0:
        ax1.scatter(constrained_relative_coordinates[:, 0],
                    constrained_relative_coordinates[:, 1],
                    c='green',
                    marker='x',
                    linewidths=2,
                    alpha=0.5,
                    label='Constraint')

    first = True
    for center in reference_relative_coordinates:
        for i in [1, 2, 3]:
            if first:
                circle_label = f'{i}' + r'$\sigma_d$'
            else:
                circle_label = '__nolegend__'
            circle = Circle(center, i * sigma_d,
                            facecolor='blue',
                            edgecolor='blue',
                            alpha=0.5 / i,
                            linewidth=1,
                            label=circle_label
                            )
            ax1.add_patch(circle)
        first = False

    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left')
    fig1.tight_layout()
    return fig1


def get_ffmpeg_command(images_directory: Path, video_output_path: Path):
    """Create a command to drive the creation of a mpeg video from many images written on disk."""
    commands = [
        "ffmpeg",
        "-r",
        "10",
        "-start_number",
        "0",
        "-i",
        str(images_directory / "frame_%d.png"),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_output_path),
    ]
    return commands


def create_2d_trajectory_video(trajectory: torch.Tensor,
                               reference_relative_coordinates: torch.Tensor,
                               constrained_relative_coordinates: torch.Tensor,
                               sigma_d: float,
                               video_output_path: Path):
    """Create a video of a 2D trajectory."""
    images_directory = video_output_path.parent / "frames"
    images_directory.mkdir(parents=True, exist_ok=True)

    for frame_idx, relative_coordinates in tqdm(enumerate(trajectory), "VIDEO"):
        fig = plot_2d_relative_coordinates(relative_coordinates,
                                           reference_relative_coordinates,
                                           constrained_relative_coordinates,
                                           sigma_d)
        fig.savefig(images_directory / f"frame_{frame_idx}.png")
        plt.close(fig)

    command = get_ffmpeg_command(images_directory, video_output_path)

    # This subprocess should create the video.
    subprocess.run(command, capture_output=True, text=True)

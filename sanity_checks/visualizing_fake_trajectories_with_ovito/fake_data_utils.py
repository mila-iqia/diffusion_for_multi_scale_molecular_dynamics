from pathlib import Path

import einops
import torch

from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from crystal_diffusion.utils.sample_trajectory import ODESampleTrajectory


def generate_fake_trajectories_pickle(
    acell: float,
    number_of_atoms: int,
    number_of_frames: int,
    number_of_trajectories: int,
    pickle_path: Path,
):
    """Generate fake trajectories pickle.

    This function creates a torch pickle with the needed data fields to sanity check that
    we can visualize paths with Ovito.

    Args:
        acell : A cell dimension parameter
        number_of_atoms :  number of atoms in the cell
        number_of_frames : number of time steps in the trajectories
        number_of_trajectories : number of trajectories
        pickle_path : where the pickle should be written

    Returns:
        None.
    """
    spatial_dimension = 3
    t0 = 0.0
    tf = 1.0
    # These parameters don't really matter for the purpose of generating fake trajectories.
    sigma_min = 0.01
    sigma_max = 0.5

    # Times have dimension [number_of_time_steps]
    times = torch.linspace(tf, t0, number_of_frames)

    # evaluation_times have dimension [batch_size, number_of_time_steps]
    evaluation_times = einops.repeat(
        times, "t -> batch t", batch=number_of_trajectories
    )

    shifts = torch.rand(number_of_trajectories)

    a = acell + torch.cos(2 * torch.pi * shifts)
    b = acell + torch.sin(2 * torch.pi * shifts)
    c = acell + shifts

    # unit_cells have dimensions [number_of_samples, spatial_dimension, spatial_dimension]
    unit_cells = torch.diag_embed(einops.rearrange([a, b, c], "d batch -> batch d"))

    sigmas = sigma_min ** (1.0 - evaluation_times) * sigma_max**evaluation_times

    normalized_scores = 0.1 * torch.rand(
        number_of_trajectories, number_of_frames, number_of_atoms, spatial_dimension
    )

    initial_relative_coordinates = torch.rand(
        [number_of_trajectories, 1, number_of_atoms, spatial_dimension]
    )
    relative_coordinates = map_relative_coordinates_to_unit_cell(
        initial_relative_coordinates + normalized_scores
    )

    sample_trajectory_recorder = ODESampleTrajectory()

    sample_trajectory_recorder.record_unit_cell(unit_cells)

    sample_trajectory_recorder.record_ode_solution(
        times=evaluation_times,
        sigmas=sigmas,
        relative_coordinates=relative_coordinates,
        normalized_scores=normalized_scores,
        stats="not applicable",
        status="not applicable",
    )

    sample_trajectory_recorder.write_to_pickle(pickle_path)

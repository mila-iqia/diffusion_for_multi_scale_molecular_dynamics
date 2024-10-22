"""Diffusion sample position analysis.

A simple script to extract the diffusion positions from a pickle on disk and generate
some distributions.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from crystal_diffusion import TOP_DIR
from crystal_diffusion.utils.sample_trajectory import \
    PredictorCorrectorSampleTrajectory
from src.crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH

plt.style.use(PLOT_STYLE_PATH)


# Hard Coding parameters that match some local data. Modify as needed...
epoch = 8
experiment_name = "mace_plus_prediction_head/run1"
# experiment_name = "mlp/run1"
# experiment_name = "mace/run1/"


trajectory_top_output_directory = TOP_DIR / "experiments/diffusion_mace_harmonic_data/output/"

trajectory_data_directory = trajectory_top_output_directory / experiment_name / "diffusion_position_samples"
energy_sample_directory = trajectory_top_output_directory / experiment_name / "energy_samples"


output_top_dir = trajectory_data_directory.parent / "visualization"

if __name__ == '__main__':

    energies = torch.load(energy_sample_directory / f"energies_sample_epoch={epoch}.pt")

    pickle_path = trajectory_data_directory / f"diffusion_position_sample_epoch={epoch}.pt"
    sample_trajectory = PredictorCorrectorSampleTrajectory.read_from_pickle(pickle_path)

    pickle_path = trajectory_data_directory / f"diffusion_position_sample_epoch={epoch}.pt"
    sample_trajectory = PredictorCorrectorSampleTrajectory.read_from_pickle(pickle_path)

    list_predictor_coordinates = sample_trajectory.data['predictor_x_i']
    float_datatype = list_predictor_coordinates[0].dtype

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f"Experiment {experiment_name}, Epoch = {epoch}")
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(234)
    ax4 = fig.add_subplot(235)
    ax5 = fig.add_subplot(236)
    ax1.set_title("Displacement Norm Distribution")
    ax2.set_title("Angle with (1,1,1) Direction")

    nbins = 50
    list_t = np.arange(len(list_predictor_coordinates))
    list_indices = [0, 40, 80, 99]

    equilibrium_delta_squared = 0.75
    default_orientation = torch.tensor([1., 1., 1.], dtype=float_datatype) / np.sqrt(3.)
    for idx in list_indices:
        time_idx = list_t[idx]
        relative_coordinates = list_predictor_coordinates[idx]
        displacements = relative_coordinates[:, 1, :] - relative_coordinates[:, 0, :]
        center_of_mass = 0.5 * (relative_coordinates[:, 1, :] + relative_coordinates[:, 0, :])
        delta_squared = (displacements ** 2).sum(dim=1)

        normalized_angle = torch.arccos(displacements @ default_orientation / torch.sqrt(delta_squared)) / (np.pi)

        ax1.hist(delta_squared, bins=nbins, alpha=0.5, histtype="stepfilled", label=f't = {time_idx}')
        ax2.hist(normalized_angle, bins=nbins, alpha=0.5, histtype="stepfilled", label=f't = {time_idx}')

        ax3.hist(center_of_mass[:, 0], bins=nbins, alpha=0.5, histtype="stepfilled", label=f't = {time_idx}')
        ax4.hist(center_of_mass[:, 1], bins=nbins, alpha=0.5, histtype="stepfilled", label=f't = {time_idx}')
        ax5.hist(center_of_mass[:, 2], bins=nbins, alpha=0.5, histtype="stepfilled", label=f't = {time_idx}')

    ax1.vlines(x=equilibrium_delta_squared, ymin=0, ymax=75,
               linestyle='--', color='black', label='Equilibrium Distance')
    ax2.vlines(x=torch.arccos(torch.tensor([-1, -1 / 3, 1 / 3, 1])) / np.pi, ymin=0, ymax=180,
               linestyle='--', color='black', label='Equilibrium Angles')

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_ylabel('Counts')

    ax1.legend(loc=0, fontsize=8)
    ax1.set_xlabel('$\\delta^2$')
    ax2.set_xlabel('Angle ($\\pi$)')
    ax3.set_xlabel('CM x direction')
    ax4.set_xlabel('CM y direction')
    ax5.set_xlabel('CM z direction')

    ax1.set_xlim(xmin=-0.01)
    for ax in [ax2, ax3, ax4, ax5]:
        ax.set_xlim(-0.01, 1.01)

    fig.subplots_adjust(hspace=0.4)
    plt.show()

import glob
from pathlib import Path

import einops
import torch
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.analysis import \
    PLOT_STYLE_PATH, PLEASANT_FIG_SIZE
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL, NOISY_AXL_COMPOSITION, NOISE, TIME, UNIT_CELL, \
    CARTESIAN_FORCES
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import VarianceScheduler
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.sample_diffusion import \
    get_axl_network
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from experiments.three_atoms_in_one_dimension import RESULTS_DIR, EXPERIMENTS_DIR

plt.style.use(PLOT_STYLE_PATH)

def get_batch(relative_coordinates, sigma, time):
    forces = torch.zeros_like(relative_coordinates)
    batch_size, natoms, _ = relative_coordinates.shape

    atom_types = torch.ones(batch_size, natoms, dtype=torch.int64)

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
    return batch


output_file_path = RESULTS_DIR / "egnn.mp4"
checkpoint_path = glob.glob(str(EXPERIMENTS_DIR / "**/last_model*.ckpt"), recursive=True)[0]

sigma_min = 0.001
sigma_max = 0.2
sigma_d = 0.01

spatial_dimension = 1
x0_1 = 1.0 / 3.0
x0_2 = 2.0 / 3.0
x0_3 = 0.0

#model = "analytical"
model = "learned"

if __name__ == "__main__":
    checkpoint_name = Path(checkpoint_path).name
    axl_network = get_axl_network(checkpoint_path)

    noise_parameters = NoiseParameters(
        total_time_steps=10, sigma_min=sigma_min, sigma_max=sigma_max, schedule_type= "linear"
    )

    score_network_parameters = AnalyticalScoreNetworkParameters(
        number_of_atoms=3,
        num_atom_types=1,
        kmax=5,
        equilibrium_relative_coordinates=[[x0_1], [x0_2], [x0_3]],
        sigma_d=sigma_d,
        spatial_dimension=spatial_dimension,
        use_permutation_invariance=True,
    )

    analytical_score_network = AnalyticalScoreNetwork(score_network_parameters)

    time = 0.5
    sigma = VarianceScheduler(noise_parameters).get_sigma(torch.tensor([time])).item()

    """
    trajectory_relative_coordinates = torch.linspace(0, 0.2, 1001).repeat(3, 1).transpose(1, 0).unsqueeze(-1)
    trajectory_batch = get_batch(trajectory_relative_coordinates, sigma, time)

    with torch.no_grad():
        trajectory_analytical_score = analytical_score_network(trajectory_batch).X.squeeze(-1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(trajectory_analytical_score[:, 0], lw=4)
    ax.plot(trajectory_analytical_score[:, 1], lw=2)
    ax.plot(trajectory_analytical_score[:, 2], lw=1)
    plt.show()
    """


    initial_point = torch.tensor([0., 0., 0.])
    #final_point = torch.tensor([2., 1., 0.])
    final_point = torch.tensor([1., 0., -1.])
    #final_point = torch.tensor([1.5, 0.5, -0.5])
    direction = (final_point - initial_point)
    normalized_direction = direction / torch.norm(direction)

    list_d = torch.linspace(0, 1, 1001)
    trajectory_relative_coordinates = map_relative_coordinates_to_unit_cell(
        (list_d.unsqueeze(1) * direction).unsqueeze(-1))

    trajectory_batch = get_batch(trajectory_relative_coordinates, sigma, time)

    with torch.no_grad():
        trajectory_analytical_score = analytical_score_network(trajectory_batch).X.squeeze(-1)
        trajectory_model_score = axl_network(trajectory_batch).X.squeeze(-1)

    #p1 = trajectory_analytical_score[:, 0] - trajectory_analytical_score[:, 1]
    #p2 = trajectory_analytical_score[:, 1] - trajectory_analytical_score[:, 2]
    #projected_analytical_score = (p1 + p2)/ torch.sqrt(torch.tensor(2))
    projected_analytical_score = torch.matmul(trajectory_analytical_score, normalized_direction)

    #projected_model_score = torch.matmul(trajectory_model_score, normalized_direction)
    #p1 = trajectory_model_score[:, 0] - trajectory_model_score[:, 1]
    #p2 = trajectory_model_score[:, 1] - trajectory_model_score[:, 2]
    #projected_model_score = (p1 + p2)/ torch.sqrt(torch.tensor(2))

    projected_model_score = torch.matmul(trajectory_model_score, normalized_direction)

    title = r"Normalized Score Along Trajectory, $\sigma$ = {:4.3f}".format(sigma)
    fig0 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig0.suptitle(title)
    ax0 = fig0.add_subplot(111)
    ax0.plot(list_d, projected_analytical_score, "g-", label='Analytical')
    ax0.plot(list_d, projected_model_score, "r-", label='Model')
    ax0.spines["top"].set_visible(True)
    ax0.spines["right"].set_visible(True)

    ax0.set_xlabel("$d$")
    ax0.set_ylabel(r"$\bf s$")
    ax0.set_xlim(-0.01, 1.01)
    #ax0.set_ylim(-0.01, 1.01)
    fig0.tight_layout()
    plt.show()



    number_of_spatial_points = 50  # let's be opinionated about this.
    n1 = number_of_spatial_points
    n2 = number_of_spatial_points

    u = torch.linspace(0.0, 1.0, n1)
    v = torch.linspace( 0.0, 1.0, n2)

    U, V_ = torch.meshgrid(u, v, indexing="xy")
    V = torch.flip(V_, dims=[0])

    X1 = (2 * U + V) / 3.
    X2 = (-U + V) / 3.
    X3 = (-U - 2 * V) / 3.

    relative_coordinates = einops.repeat(
        [X1, X2, X3], "natoms n1 n2 -> (n1 n2) natoms space", space=spatial_dimension
    ).contiguous()
    relative_coordinates = map_relative_coordinates_to_unit_cell(relative_coordinates)

    batch = get_batch(relative_coordinates, sigma, time)

    if model == "learned":
        model_predictions = axl_network(batch)
    elif model == "analytical":
        model_predictions = analytical_score_network(batch)

    sigma_normalized_scores = einops.rearrange(
        model_predictions.X.detach(),
        "(n1 n2) natoms 1 -> n1 n2 natoms",
        n1=n1,
        n2=n2,
    )

    S1 = sigma_normalized_scores[:, :, 0].numpy()
    S2 = sigma_normalized_scores[:, :, 1].numpy()
    S3 = sigma_normalized_scores[:, :, 2].numpy()

    N1 = S1 - S2
    N2 = S2 - S3
    N3 = S1 + S2 + S3

    title = (r"{} model: $\sigma$ = {:4.3f}, Max|$\sigma s$| = {:3.2f}"
             .format(model, sigma, sigma_normalized_scores.abs().max()))

    fig1 = plt.figure(figsize=(0.9 * PLEASANT_FIG_SIZE[0], 0.9 * PLEASANT_FIG_SIZE[0]))
    fig1.suptitle(title)
    ax1 = fig1.add_subplot(111)
    _ = ax1.quiver(U, V, N1, N2, units="width", color="r")
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)

    ax1.set_xlabel("$x_1 - x_2$")
    ax1.set_ylabel("$x_2 - x_3$")
    ax1.set_xlim(-0.01, 1.01)
    ax1.set_ylim(-0.01, 1.01)

    fig1.tight_layout()
    plt.show()

    figsize = (2.5 * PLEASANT_FIG_SIZE[0], 0.75 * PLEASANT_FIG_SIZE[0])
    fig2 = plt.figure(figsize=figsize)
    fig2.suptitle(title)
    ax2 = fig2.add_subplot(131)
    ax2.set_title("$s_1 - s_2$")
    ax3 = fig2.add_subplot(132)
    ax3.set_title("$s_2 - s_3$")
    ax4 = fig2.add_subplot(133)
    ax4.set_title("$s_1 + s_2 + s_3$")

    pos2 = ax2.imshow(N1, cmap='jet', interpolation="bicubic")
    fig2.colorbar(pos2, ax=ax2)

    pos3 = ax3.imshow(N2, cmap='jet', interpolation="bicubic")
    fig2.colorbar(pos3, ax=ax3)

    pos4 = ax4.imshow(N3, cmap='jet', interpolation="bicubic")
    fig2.colorbar(pos4, ax=ax4)

    for ax in [ax2, ax3, ax4]:
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        ax.set_xlabel("$x_1 - x_2$")
        ax.set_ylabel("$x_2 - x_3$")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    fig2.tight_layout()
    plt.show()

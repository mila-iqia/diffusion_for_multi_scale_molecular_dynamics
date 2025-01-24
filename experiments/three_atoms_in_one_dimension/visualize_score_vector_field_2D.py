import glob
from pathlib import Path

import einops
import torch
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.equivariant_analytical_score_network import (
    EquivariantAnalyticalScoreNetwork,
    EquivariantAnalyticalScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import \
    VarianceScheduler
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.sample_diffusion import \
    get_axl_network
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from experiments.three_atoms_in_one_dimension import (EXPERIMENTS_DIR,
                                                      RESULTS_DIR)

plt.style.use(PLOT_STYLE_PATH)


def get_batch(relative_coordinates, sigma, time):
    """Get batch."""
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


checkpoint_path = glob.glob(
    str(EXPERIMENTS_DIR / "**/last_model*.ckpt"), recursive=True
)[0]

sigma_min = 0.001
sigma_max = 0.2
sigma_d = 0.01

spatial_dimension = 1
x0_1 = 1.0 / 3.0
x0_2 = 2.0 / 3.0
x0_3 = 0.0

if __name__ == "__main__":

    checkpoint_name = Path(checkpoint_path).name
    axl_network = get_axl_network(checkpoint_path)

    noise_parameters = NoiseParameters(
        total_time_steps=10,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        schedule_type="linear",
    )

    analytical_model_common_parameters = dict(
        number_of_atoms=3,
        num_atom_types=1,
        kmax=5,
        equilibrium_relative_coordinates=[[x0_1], [x0_2], [x0_3]],
        sigma_d=sigma_d,
        spatial_dimension=spatial_dimension,
    )

    params = AnalyticalScoreNetworkParameters(
        **analytical_model_common_parameters, use_permutation_invariance=True
    )
    analytical_score_network = AnalyticalScoreNetwork(params)

    params = EquivariantAnalyticalScoreNetworkParameters(
        **analytical_model_common_parameters
    )
    equivariant_analytical_score_network = EquivariantAnalyticalScoreNetwork(params)

    initial_point = torch.tensor([0.0, 0.0, 0.0])
    final_point = torch.tensor([1.0, 0.0, -1.0])
    direction = final_point - initial_point
    normalized_direction = direction / torch.norm(direction)

    list_d = torch.linspace(0, 1, 1001)
    trajectory_relative_coordinates = map_relative_coordinates_to_unit_cell(
        (list_d.unsqueeze(1) * direction).unsqueeze(-1)
    )

    times = torch.tensor([0.0, 0.25, 0.5, 1.0])
    sigmas = VarianceScheduler(noise_parameters).get_sigma(times)

    title = r"Normalized Score Along Trajectory"
    fig0 = plt.figure(figsize=(2 * PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[1]))
    fig0.suptitle(title)

    ax1 = fig0.add_subplot(141)
    ax2 = fig0.add_subplot(142)
    ax3 = fig0.add_subplot(143)
    ax4 = fig0.add_subplot(144)
    list_ax = [ax1, ax2, ax3, ax4]

    for time, sigma, ax in zip(times, sigmas, list_ax):
        ax.set_title(r"$t$ = {:2.1f}, $\sigma$ = {:4.3f}".format(time, sigma))
        trajectory_batch = get_batch(trajectory_relative_coordinates, sigma, time)

        with torch.no_grad():
            trajectory_analytical_score = analytical_score_network(
                trajectory_batch
            ).X.squeeze(-1)
            trajectory_equivariant_analytical_score = (
                equivariant_analytical_score_network(trajectory_batch).X.squeeze(-1)
            )
            trajectory_model_score = axl_network(trajectory_batch).X.squeeze(-1)

        projected_analytical_score = torch.matmul(
            trajectory_analytical_score, normalized_direction
        )
        projected_equivariant_analytical_score = torch.matmul(
            trajectory_equivariant_analytical_score, normalized_direction
        )
        projected_model_score = torch.matmul(
            trajectory_model_score, normalized_direction
        )

        ax.plot(
            list_d,
            projected_analytical_score,
            "g-",
            lw=4,
            alpha=0.5,
            label="Analytical",
        )
        ax.plot(
            list_d,
            projected_equivariant_analytical_score,
            "b--",
            lw=2,
            label="Equivariant Analytical",
        )
        ax.plot(list_d, projected_model_score, "r-", label="Learned Model")

        ax.legend(loc=0)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        ax.set_xlabel("$d$")
        ax.set_ylabel(r"$\bf s$")
        ax.hlines(0, -0.01, 1.01, linestyles="-", color="grey", alpha=0.5)
        ax.set_xlim(-0.01, 1.01)

    fig0.tight_layout()
    fig0.savefig(RESULTS_DIR / "scores_along_trajectory.png")

    # ================================================================================
    time = torch.tensor([0.5])
    sigma = VarianceScheduler(noise_parameters).get_sigma(time)

    number_of_spatial_points = 50  # let's be opinionated about this.
    n1 = number_of_spatial_points
    n2 = number_of_spatial_points

    u = torch.linspace(0.0, 1.0, n1)
    v = torch.linspace(0.0, 1.0, n2)

    U, V_ = torch.meshgrid(u, v, indexing="xy")
    V = torch.flip(V_, dims=[0])

    X1 = (2 * U + V) / 3.0
    X2 = (-U + V) / 3.0
    X3 = (-U - 2 * V) / 3.0

    relative_coordinates = einops.repeat(
        [X1, X2, X3], "natoms n1 n2 -> (n1 n2) natoms space", space=spatial_dimension
    ).contiguous()
    relative_coordinates = map_relative_coordinates_to_unit_cell(relative_coordinates)

    batch = get_batch(relative_coordinates, sigma, time)

    list_models = ["learned", "analytical", "equivariant_analytical"]
    list_score_networks = [
        axl_network,
        analytical_score_network,
        equivariant_analytical_score_network,
    ]

    for model, score_network in zip(list_models, list_score_networks):
        model_predictions = score_network(batch)

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

        max_s = sigma_normalized_scores.abs().max().item()
        title = r"{} model: $\sigma$ = {:4.3f}, Max|$\sigma s$| = {:3.2f}".format(
            model, sigma.item(), max_s
        )

        fig1 = plt.figure(
            figsize=(0.9 * PLEASANT_FIG_SIZE[0], 0.9 * PLEASANT_FIG_SIZE[0])
        )
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
        fig1.savefig(RESULTS_DIR / f"vector_field_{model}_model.png")

        extent = (0.0, 1.0, 0.0, 1.0)
        figsize = (2.5 * PLEASANT_FIG_SIZE[0], 0.75 * PLEASANT_FIG_SIZE[0])
        fig2 = plt.figure(figsize=figsize)
        fig2.suptitle(title)
        ax2 = fig2.add_subplot(131)
        ax2.set_title("$s_1 - s_2$")
        ax3 = fig2.add_subplot(132)
        ax3.set_title("$s_2 - s_3$")
        ax4 = fig2.add_subplot(133)
        ax4.set_title("$s_1 + s_2 + s_3$")

        pos2 = ax2.imshow(N1, cmap="jet", interpolation="bicubic", extent=extent)
        fig2.colorbar(pos2, ax=ax2)

        pos3 = ax3.imshow(N2, cmap="jet", interpolation="bicubic", extent=extent)
        fig2.colorbar(pos3, ax=ax3)

        pos4 = ax4.imshow(N3, cmap="jet", interpolation="bicubic", extent=extent)
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
        fig2.savefig(RESULTS_DIR / f"imshow_{model}_model.png")

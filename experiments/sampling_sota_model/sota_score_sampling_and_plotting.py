"""Sampling and plotting of the score coming from our SOTA model."""

import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from einops import einops

from diffusion_for_multi_scale_molecular_dynamics import DATA_DIR
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.ode_position_generator import (
    ExplodingVarianceODEAXLGenerator, ODESamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.instantiate_diffusion_model import \
    load_diffusion_model
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import \
    VarianceScheduler
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps import \
    get_energy_and_forces_from_lammps
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    setup_analysis_logger

logger = logging.getLogger(__name__)
setup_analysis_logger()

experiments_dir = Path("/home/mila/r/rousseab/experiments/")
model_dir = experiments_dir / "checkpoints/sota_model/"
state_dict_path = model_dir / "last_model-epoch=199-step=019600_state_dict.ckpt"
config_path = model_dir / "config_backup.yaml"

SOTA_SCORE_RESULTS_DIR = Path(
    "/home/mila/r/rousseab/experiments/draw_sota_samples/figures"
)
SOTA_SCORE_RESULTS_DIR.mkdir(exist_ok=True)


plt.style.use(PLOT_STYLE_PATH)


device = torch.device("cuda")

# Change these parameters as needed!
# sampling_algorithm = 'ode'
sampling_algorithm = "langevin"

spatial_dimension = 3
number_of_atoms = 8
atom_types = np.ones(number_of_atoms, dtype=int)

acell = 5.43
box = np.diag([acell, acell, acell])

number_of_samples = 1000
total_time_steps = 100
number_of_corrector_steps = 1

if __name__ == "__main__":

    logger.info("Loading state dict")
    with open(str(state_dict_path), "rb") as fd:
        state_dict = torch.load(fd)

    with open(str(config_path), "r") as fd:
        hyper_params = yaml.load(fd, Loader=yaml.FullLoader)
    logger.info("Instantiate model")
    pl_model = load_diffusion_model(hyper_params)
    pl_model.load_state_dict(state_dict=state_dict)
    pl_model.to(device)
    pl_model.eval()

    sigma_normalized_score_network = pl_model.sigma_normalized_score_network

    logger.info("Setting up parameters")
    noise_parameters = NoiseParameters(
        total_time_steps=total_time_steps, sigma_min=0.001, sigma_max=0.5
    )
    exploding_variance = VarianceScheduler(noise_parameters)

    if sampling_algorithm == "ode":
        ode_sampling_parameters = ODESamplingParameters(
            spatial_dimension=spatial_dimension,
            number_of_atoms=number_of_atoms,
            number_of_samples=number_of_samples,
            cell_dimensions=[acell, acell, acell],
            record_samples=True,
            absolute_solver_tolerance=1.0e-5,
            relative_solver_tolerance=1.0e-5,
        )

        position_generator = ExplodingVarianceODEAXLGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=ode_sampling_parameters,
            sigma_normalized_score_network=sigma_normalized_score_network,
        )

    elif sampling_algorithm == "langevin":
        pc_sampling_parameters = PredictorCorrectorSamplingParameters(
            number_of_corrector_steps=number_of_corrector_steps,
            spatial_dimension=spatial_dimension,
            number_of_atoms=number_of_atoms,
            number_of_samples=number_of_samples,
            cell_dimensions=[acell, acell, acell],
            record_samples=True,
        )

        position_generator = LangevinGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=pc_sampling_parameters,
            sigma_normalized_score_network=sigma_normalized_score_network,
        )

    # Draw some samples, create some plots
    unit_cells = torch.Tensor(box).repeat(number_of_samples, 1, 1).to(device)

    logger.info("Drawing samples")

    with torch.no_grad():
        samples = position_generator.sample(
            number_of_samples=number_of_samples, device=device, unit_cell=unit_cells
        )

    batch_relative_positions = samples.cpu().numpy()
    batch_positions = np.dot(batch_relative_positions, box)

    list_energy = []
    logger.info("Compute energy from Oracle")
    with tempfile.TemporaryDirectory() as lammps_work_directory:
        for idx, positions in enumerate(batch_positions):
            energy, forces = get_energy_and_forces_from_lammps(
                positions,
                box,
                atom_types,
                tmp_work_dir=lammps_work_directory,
                pair_coeff_dir=DATA_DIR,
            )
            list_energy.append(energy)
    energies = np.array(list_energy)

    if sampling_algorithm == "ode":
        # Plot the ODE parameters
        logger.info("Plotting ODE parameters")
        times = torch.linspace(0, 1, 1001)
        sigmas = exploding_variance.get_sigma(times)
        ode_prefactor = position_generator._get_ode_prefactor(sigmas)

        fig0 = plt.figure(figsize=PLEASANT_FIG_SIZE)
        fig0.suptitle("ODE parameters")

        ax1 = fig0.add_subplot(121)
        ax2 = fig0.add_subplot(122)
        ax1.set_title("$\\sigma$ Parameter")
        ax2.set_title("$\\gamma$ Parameter")
        ax1.plot(times, sigmas, "-")
        ax2.plot(times, ode_prefactor, "-")

        ax1.set_ylabel("$\\sigma(t)$")
        ax2.set_ylabel("$\\gamma(t)$")
        for ax in [ax1, ax2]:
            ax.set_xlabel("Diffusion Time")
            ax.set_xlim([-0.01, 1.01])

        fig0.tight_layout()
        fig0.savefig(SOTA_SCORE_RESULTS_DIR / "ODE_parameters.png")
        plt.close(fig0)

    logger.info("Extracting data artifacts")
    raw_data = position_generator.sample_trajectory_recorder.data
    recorded_data = position_generator.sample_trajectory_recorder.standardize_data(
        raw_data
    )

    sampling_times = recorded_data["time"].cpu()
    relative_coordinates = recorded_data["relative_coordinates"]
    batch_flat_relative_coordinates = einops.rearrange(
        relative_coordinates, "b t n d -> b t (n d)"
    )

    normalized_scores = recorded_data["normalized_scores"]
    batch_flat_normalized_scores = einops.rearrange(
        normalized_scores, "b t n d -> b t (n d)"
    )

    # ============================================================================
    logger.info("Plotting relative coordinates trajectories")
    fig1 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig1.suptitle("Sampling Trajectories")

    ax = fig1.add_subplot(111)
    ax.set_xlabel("Diffusion Time")
    ax.set_ylabel("Raw Relative Coordinate")
    ax.yaxis.tick_right()
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    alpha = 1.0
    for flat_relative_coordinates in batch_flat_relative_coordinates[::100]:
        for i in range(number_of_atoms * spatial_dimension):
            coordinate = flat_relative_coordinates[:, i].cpu()
            ax.plot(sampling_times, coordinate, "-", color="b", alpha=alpha)
        alpha = 0.05

    ax.set_xlim([1.01, -0.01])
    fig1.savefig(
        SOTA_SCORE_RESULTS_DIR
        / f"sampling_trajectories_{sampling_algorithm}_{number_of_atoms}_atoms.png"
    )
    plt.close(fig1)

    # ========================   Figure 2   ======================================
    logger.info("Plotting scores along trajectories")
    fig2 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig2.suptitle("Root Mean Squared Normalized Scores Along Sample Trajectories")
    rms_norm_score = (batch_flat_normalized_scores**2).mean(dim=-1).sqrt().cpu().numpy()

    ax1 = fig2.add_subplot(121)
    ax2 = fig2.add_subplot(122)
    for y in rms_norm_score[::10]:
        for ax in [ax1, ax2]:
            ax.plot(
                sampling_times, y, "-", color="gray", alpha=0.2, label="__nolabel__"
            )

    list_quantiles = [0.0, 0.10, 0.5, 1.0]
    list_colors = ["green", "yellow", "orange", "red"]

    for q, c in zip(list_quantiles, list_colors):
        energy_quantile = np.quantile(energies, q)
        idx = np.argmin(np.abs(energies - energy_quantile))
        e = energies[idx]
        for ax in [ax1, ax2]:
            ax.plot(
                sampling_times,
                rms_norm_score[idx],
                "-",
                color=c,
                alpha=1.0,
                label=f"{100 * q:2.0f}% Percentile Energy: {e:5.1f}",
            )

    for ax in [ax1, ax2]:
        ax.set_xlabel("Diffusion Time")
        ax.set_ylabel(r"$ \sqrt{\langle (\sigma(t) S_{\theta} )^2\rangle}$")
        ax.set_xlim(1, 0)

    ax1.legend(loc=0, fontsize=6)
    ax1.set_yscale("log")

    fig2.savefig(
        SOTA_SCORE_RESULTS_DIR
        / f"sampling_score_trajectories_{sampling_algorithm}_{number_of_atoms}_atoms.png"
    )
    plt.close(fig2)

    # ========================   Figure 3   ======================================
    logger.info("Plotting Marginal distribution in 2D")
    fig3 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig3.suptitle("Sampling Marginal Distributions")
    ax1 = fig3.add_subplot(131, aspect="equal")
    ax2 = fig3.add_subplot(132, aspect="equal")
    ax3 = fig3.add_subplot(133, aspect="equal")

    xs = einops.rearrange(samples, "b n d -> (b n) d").cpu()
    ax1.set_title("XY Projection")
    ax1.plot(xs[:, 0], xs[:, 1], "ro", alpha=0.5, mew=0, label="ODE Solver")

    ax2.set_title("XZ Projection")
    ax2.plot(xs[:, 0], xs[:, 2], "ro", alpha=0.5, mew=0, label="ODE Solver")

    ax3.set_title("YZ Projection")
    ax3.plot(xs[:, 1], xs[:, 2], "ro", alpha=0.5, mew=0, label="ODE Solver")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.vlines(x=[0, 1], ymin=0, ymax=1, color="k", lw=2)
        ax.hlines(y=[0, 1], xmin=0, xmax=1, color="k", lw=2)

    ax1.legend(loc=0)
    fig3.tight_layout()
    fig3.savefig(
        SOTA_SCORE_RESULTS_DIR
        / f"marginal_2D_distributions_{sampling_algorithm}_{number_of_atoms}_atoms.png"
    )
    plt.close(fig3)

    # ========================   Figure 4   ======================================
    logger.info("Plotting Marginal distribution in 1D")
    fig4 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    ax1 = fig4.add_subplot(131)
    ax2 = fig4.add_subplot(132)
    ax3 = fig4.add_subplot(133)
    fig4.suptitle("Comparing Sampling and Expected Marginal Distributions")

    common_params = dict(histtype="stepfilled", alpha=0.5, bins=50)

    ax1.hist(xs[:, 0], **common_params, facecolor="r")
    ax2.hist(xs[:, 1], **common_params, facecolor="r")
    ax3.hist(xs[:, 2], **common_params, facecolor="r")

    ax1.set_xlabel("X")
    ax2.set_xlabel("Y")
    ax3.set_xlabel("Z")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-0.01, 1.01)

    fig4.tight_layout()
    fig4.savefig(
        SOTA_SCORE_RESULTS_DIR
        / f"marginal_1D_distributions_{sampling_algorithm}_{number_of_atoms}_atoms.png"
    )
    plt.close(fig4)

    # ========================   Figure 5   ======================================
    logger.info("Plotting energy distributions")
    fig5 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig5.suptitle(f"Energy Distribution, sampling algorithm {sampling_algorithm}")

    common_params = dict(density=True, bins=50, histtype="stepfilled", alpha=0.25)

    ax1 = fig5.add_subplot(111)
    ax1.hist(energies, **common_params, label="Sampled Energies", color="red")

    ax1.set_xlabel("Energy (eV)")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right", fancybox=True, shadow=True, ncol=1, fontsize=12)
    fig5.tight_layout()
    fig5.savefig(
        SOTA_SCORE_RESULTS_DIR
        / f"energy_samples_{sampling_algorithm}_{number_of_atoms}_atoms.png"
    )
    plt.close(fig5)

    logger.info("Done!")

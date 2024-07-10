"""Sampling and plotting of the Analytical score.

This little ad hoc experiment explores sampling using the 'analytic' score,
plotting various artifacts along the way.

It shows that both ODE and Langevin sampling work very well when the score is good!
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import einops

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.analysis.analytic_score import \
    ANALYTIC_SCORE_RESULTS_DIR
from crystal_diffusion.analysis.analytic_score.utils import (
    get_exact_samples, get_samples_harmonic_energy, get_silicon_supercell,
    get_unit_cells)
from crystal_diffusion.generators.langevin_generator import LangevinGenerator
from crystal_diffusion.generators.ode_position_generator import (
    ExplodingVarianceODEPositionGenerator, ODESamplingParameters)
from crystal_diffusion.generators.predictor_corrector_position_generator import \
    PredictorCorrectorSamplingParameters
from crystal_diffusion.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.logging_utils import setup_analysis_logger

logger = logging.getLogger(__name__)
setup_analysis_logger()


plt.style.use(PLOT_STYLE_PATH)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Change these parameters as needed!
sampling_algorithm = 'ode'
# sampling_algorithm = 'langevin'
supercell_factor = 1

kmax = 1

variance_parameter = 0.001 / supercell_factor
number_of_samples = 1000
total_time_steps = 101
number_of_corrector_steps = 1

if __name__ == '__main__':

    logger.info("Setting up parameters")
    equilibrium_relative_coordinates = torch.from_numpy(
        get_silicon_supercell(supercell_factor=supercell_factor)).to(device)
    number_of_atoms, spatial_dimension = equilibrium_relative_coordinates.shape
    nd = number_of_atoms * spatial_dimension

    noise_parameters = NoiseParameters(total_time_steps=total_time_steps,
                                       sigma_min=0.001,
                                       sigma_max=0.5)

    score_network_parameters = AnalyticalScoreNetworkParameters(
        number_of_atoms=number_of_atoms,
        spatial_dimension=spatial_dimension,
        kmax=kmax,
        equilibrium_relative_coordinates=equilibrium_relative_coordinates,
        variance_parameter=variance_parameter)

    sigma_normalized_score_network = AnalyticalScoreNetwork(score_network_parameters)

    if sampling_algorithm == 'ode':
        ode_sampling_parameters = ODESamplingParameters(spatial_dimension=spatial_dimension,
                                                        number_of_atoms=number_of_atoms,
                                                        number_of_samples=number_of_samples,
                                                        cell_dimensions=[1., 1., 1.],
                                                        record_samples=True,
                                                        absolute_solver_tolerance=1.0e-5,
                                                        relative_solver_tolerance=1.0e-5)

        position_generator = (
            ExplodingVarianceODEPositionGenerator(noise_parameters=noise_parameters,
                                                  sampling_parameters=ode_sampling_parameters,
                                                  sigma_normalized_score_network=sigma_normalized_score_network))

    elif sampling_algorithm == 'langevin':
        pc_sampling_parameters = PredictorCorrectorSamplingParameters(
            number_of_corrector_steps=number_of_corrector_steps,
            spatial_dimension=spatial_dimension,
            number_of_atoms=number_of_atoms,
            number_of_samples=number_of_samples,
            cell_dimensions=[1., 1., 1.],
            record_samples=True)

        position_generator = LangevinGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=pc_sampling_parameters,
            sigma_normalized_score_network=sigma_normalized_score_network)

    # Draw some samples, create some plots
    unit_cell = get_unit_cells(acell=1.,
                               spatial_dimension=spatial_dimension,
                               number_of_samples=number_of_samples).to(device)

    logger.info("Drawing samples")
    samples = position_generator.sample(number_of_samples=number_of_samples,
                                        device=device,
                                        unit_cell=unit_cell).detach()

    if sampling_algorithm == 'ode':
        # Plot the ODE parameters
        logger.info("Plotting ODE parameters")
        times = torch.linspace(0, 1, 1001)
        sigmas = position_generator._get_exploding_variance_sigma(times)
        ode_prefactor = position_generator._get_ode_prefactor(sigmas)

        fig0 = plt.figure(figsize=PLEASANT_FIG_SIZE)
        fig0.suptitle('ODE parameters')

        ax1 = fig0.add_subplot(121)
        ax2 = fig0.add_subplot(122)
        ax1.set_title('$\\sigma$ Parameter')
        ax2.set_title('$\\gamma$ Parameter')
        ax1.plot(times, sigmas, '-')
        ax2.plot(times, ode_prefactor, '-')

        ax1.set_ylabel('$\\sigma(t)$')
        ax2.set_ylabel('$\\gamma(t)$')
        for ax in [ax1, ax2]:
            ax.set_xlabel('Diffusion Time')
            ax.set_xlim([-0.01, 1.01])

        fig0.tight_layout()
        fig0.savefig(ANALYTIC_SCORE_RESULTS_DIR / "ODE_parameters.png")
        plt.close(fig0)

    logger.info("Extracting data artifacts")
    raw_data = position_generator.sample_trajectory_recorder.data
    recorded_data = position_generator.sample_trajectory_recorder.standardize_data(raw_data)

    sampling_times = recorded_data['time']
    relative_coordinates = recorded_data['relative_coordinates']
    batch_flat_relative_coordinates = einops.rearrange(relative_coordinates, "b t n d -> b t (n d)")

    normalized_scores = recorded_data['normalized_scores']
    batch_flat_normalized_scores = einops.rearrange(normalized_scores, "b t n d -> b t (n d)")

    # ============================================================================

    logger.info("Creating samples from the exact distribution")
    inverse_covariance = (torch.diag(torch.ones(nd)) / variance_parameter).to(equilibrium_relative_coordinates)
    inverse_covariance = inverse_covariance.reshape(number_of_atoms, spatial_dimension,
                                                    number_of_atoms, spatial_dimension)
    exact_samples = get_exact_samples(equilibrium_relative_coordinates,
                                      inverse_covariance,
                                      number_of_samples).cpu()

    logger.info("Computing harmonic energies")
    exact_harmonic_energies = get_samples_harmonic_energy(equilibrium_relative_coordinates,
                                                          inverse_covariance,
                                                          exact_samples)

    sampled_harmonic_energies = get_samples_harmonic_energy(equilibrium_relative_coordinates,
                                                            inverse_covariance,
                                                            samples)

    # ========================   Figure 1   ======================================
    logger.info("Plotting relative coordinates trajectories")
    fig1 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig1.suptitle('Sampling Trajectories')

    ax = fig1.add_subplot(111)
    ax.set_xlabel('Diffusion Time')
    ax.set_ylabel('Raw Relative Coordinate')
    ax.yaxis.tick_right()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    alpha = 1.0
    for flat_relative_coordinates in batch_flat_relative_coordinates[::100]:
        for i in range(number_of_atoms * spatial_dimension):
            coordinate = flat_relative_coordinates[:, i]
            ax.plot(sampling_times.cpu(), coordinate.cpu(), '-', color='b', alpha=alpha)
        alpha = 0.05

    ax.set_xlim([1.01, -0.01])
    fig1.savefig(ANALYTIC_SCORE_RESULTS_DIR / f"sampling_trajectories_{sampling_algorithm}_{number_of_atoms}_atoms.png")
    plt.close(fig1)

    # ========================   Figure 2   ======================================
    logger.info("Plotting scores along trajectories")
    fig2 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig2.suptitle('Means Absolute Normalized Scores Along Sample Trajectories')
    mean_abs_norm_score = batch_flat_normalized_scores.abs().mean(dim=-1).numpy()

    ax1 = fig2.add_subplot(111)
    for y in mean_abs_norm_score[::10]:
        ax1.plot(sampling_times, y, '-', color='gray', alpha=0.2, label='__nolabel__')

    list_quantiles = [0.0, 0.10, 0.5, 1.0]
    list_colors = ['green', 'yellow', 'orange', 'red']

    energies = sampled_harmonic_energies.numpy()

    for q, c in zip(list_quantiles, list_colors):
        energy_quantile = np.quantile(energies, q)
        idx = np.argmin(np.abs(energies - energy_quantile))
        e = energies[idx]
        ax1.plot(sampling_times, mean_abs_norm_score[idx], '-',
                 color=c, alpha=1., label=f'{100 * q:2.0f}% Percentile Energy: {e:5.1f}')

    ax1.set_xlabel('Diffusion Time')
    ax1.set_ylabel(r'$\langle | \sigma(t) S_{\theta} | \rangle$')
    ax1.legend(loc=0)
    ax1.set_xlim(1, 0)

    fig2.savefig(
        ANALYTIC_SCORE_RESULTS_DIR / f"sampling_score_trajectories_{sampling_algorithm}_{number_of_atoms}_atoms.png")
    plt.close(fig2)

    # ========================   Figure 3   ======================================
    logger.info("Plotting Marginal distribution in 2D")
    fig3 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig3.suptitle('Comparing Sampling and Expected Marginal Distributions')
    ax1 = fig3.add_subplot(131, aspect='equal')
    ax2 = fig3.add_subplot(132, aspect='equal')
    ax3 = fig3.add_subplot(133, aspect='equal')

    xs = einops.rearrange(samples, 'b n d -> (b n) d').cpu()
    zs = einops.rearrange(exact_samples, 'b n d -> (b n) d').cpu()
    ax1.set_title('XY Projection')
    ax1.plot(xs[:, 0], xs[:, 1], 'ro', alpha=0.5, mew=0, label='ODE Solver')
    ax1.plot(zs[:, 0], zs[:, 1], 'go', alpha=0.05, mew=0, label='Exact Samples')

    ax2.set_title('XZ Projection')
    ax2.plot(xs[:, 0], xs[:, 2], 'ro', alpha=0.5, mew=0, label='ODE Solver')
    ax2.plot(zs[:, 0], zs[:, 2], 'go', alpha=0.05, mew=0, label='Exact Samples')

    ax3.set_title('YZ Projection')
    ax3.plot(xs[:, 1], xs[:, 2], 'ro', alpha=0.5, mew=0, label='ODE Solver')
    ax3.plot(zs[:, 1], zs[:, 2], 'go', alpha=0.05, mew=0, label='Exact Samples')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.vlines(x=[0, 1], ymin=0, ymax=1, color='k', lw=2)
        ax.hlines(y=[0, 1], xmin=0, xmax=1, color='k', lw=2)

    ax1.legend(loc=0)
    fig3.tight_layout()
    fig3.savefig(
        ANALYTIC_SCORE_RESULTS_DIR / f"marginal_2D_distributions_{sampling_algorithm}_{number_of_atoms}_atoms.png")
    plt.close(fig3)

    # ========================   Figure 4   ======================================
    logger.info("Plotting Marginal distribution in 1D")
    fig4 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    ax1 = fig4.add_subplot(131)
    ax2 = fig4.add_subplot(132)
    ax3 = fig4.add_subplot(133)
    fig4.suptitle('Comparing Sampling and Expected Marginal Distributions')

    common_params = dict(histtype='stepfilled', alpha=0.5, bins=50)

    ax1.hist(xs[:, 0], **common_params, facecolor='r', label='ODE solver')
    ax2.hist(xs[:, 1], **common_params, facecolor='r', label='ODE solver')
    ax3.hist(xs[:, 2], **common_params, facecolor='r', label='ODE solver')

    ax1.hist(zs[:, 0], **common_params, facecolor='g', label='Exact')
    ax2.hist(zs[:, 1], **common_params, facecolor='g', label='Exact')
    ax3.hist(zs[:, 2], **common_params, facecolor='g', label='Exact')

    ax1.set_xlabel('X')
    ax2.set_xlabel('Y')
    ax3.set_xlabel('Z')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-0.01, 1.01)

    ax1.legend(loc=0)
    fig4.tight_layout()
    fig4.savefig(
        ANALYTIC_SCORE_RESULTS_DIR / f"marginal_1D_distributions_{sampling_algorithm}_{number_of_atoms}_atoms.png")
    plt.close(fig4)

    # ========================   Figure 5   ======================================
    logger.info("Plotting harmonic energy distributions")
    fig5 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig5.suptitle(f'Energy Distribution, sampling algorithm {sampling_algorithm}')

    common_params = dict(density=True, bins=50, histtype="stepfilled", alpha=0.25)

    ax1 = fig5.add_subplot(111)
    ax1.hist(sampled_harmonic_energies, **common_params, label='Sampled Energies', color='red')
    ax1.hist(exact_harmonic_energies, **common_params, label='Theoretical Energies', color='green')

    ax1.set_xlim(xmin=-0.01)
    ax1.set_xlabel('Unitless Harmonic Energy')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper right', fancybox=True, shadow=True, ncol=1, fontsize=12)
    fig5.tight_layout()
    fig5.savefig(
        ANALYTIC_SCORE_RESULTS_DIR / f"harmonic_energy_samples_{sampling_algorithm}_{number_of_atoms}_atoms.png")
    plt.close(fig5)

    logger.info("Done!")

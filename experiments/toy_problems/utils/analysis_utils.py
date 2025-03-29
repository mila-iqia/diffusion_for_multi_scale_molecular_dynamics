import dataclasses
import glob
import os
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

import einops
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.analysis import \
    PLEASANT_FIG_SIZE
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import \
    AXL_COMPOSITION
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import \
    VarianceScheduler
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.sample_diffusion import (
    get_axl_network, main)
from diffusion_for_multi_scale_molecular_dynamics.score.wrapped_gaussian_score import \
    get_log_wrapped_gaussians

sys.path.append(str(TOP_DIR / "experiments")) # noqa

from toy_problems import EXPERIMENTS_DIR, RESULTS_DIR  # noqa
from toy_problems.utils.visualization_utils import (  # noqa
    generate_vector_field_video, plot_2d_samples)

TOY_MODEL_PARAMETERS = dict(number_of_atoms=2,
                            spatial_dimension=1,
                            num_atom_types=1,
                            cell_dimensions=[[1.0]],
                            sigma_min=0.001,
                            sigma_max=0.2,
                            schedule_type="exponential",
                            x0_1=0.25,
                            x0_2=0.75,
                            sigma_d=0.01,
                            kmax=5)

InputParameters = namedtuple("InputParameters",
                             ["algorithm", "total_time_steps", "number_of_corrector_steps",
                              "corrector_step_epsilon", "number_of_samples", "record_samples"])


def get_checkpoint_path(experiment_name: str, run_name: str):
    """Get checkpoint path."""
    if experiment_name == "analytical":
        return "analytical"

    checkpoint_top_path = EXPERIMENTS_DIR / experiment_name / "output" / run_name
    assert checkpoint_top_path.is_dir(), \
        (f"The experiment folder {checkpoint_top_path} does not exist. "
         f"Did you execute the experiment prior to trying to analyse it?")

    template_checkpoint_path = str(checkpoint_top_path / "**/best_model*.ckpt")
    checkpoint_path = glob.glob(template_checkpoint_path, recursive=True)[0]
    return checkpoint_path


def get_noise_parameters(input_parameters: InputParameters):
    """Convenience method to get noise parameters."""
    noise_parameters = NoiseParameters(
        total_time_steps=input_parameters.total_time_steps,
        sigma_min=TOY_MODEL_PARAMETERS["sigma_min"],
        sigma_max=TOY_MODEL_PARAMETERS["sigma_max"],
        schedule_type=TOY_MODEL_PARAMETERS["schedule_type"],
        corrector_step_epsilon=input_parameters.corrector_step_epsilon,
    )
    return noise_parameters


def get_analytical_score_network_parameters():
    """Convenience method to get analytical score parameters."""
    analytical_score_network_parameters = AnalyticalScoreNetworkParameters(
        number_of_atoms=TOY_MODEL_PARAMETERS["number_of_atoms"],
        num_atom_types=TOY_MODEL_PARAMETERS["num_atom_types"],
        kmax=TOY_MODEL_PARAMETERS["kmax"],
        equilibrium_relative_coordinates=[[TOY_MODEL_PARAMETERS["x0_1"]], [TOY_MODEL_PARAMETERS["x0_2"]]],
        sigma_d=TOY_MODEL_PARAMETERS["sigma_d"],
        spatial_dimension=TOY_MODEL_PARAMETERS["spatial_dimension"],
        use_permutation_invariance=True,
    )
    return analytical_score_network_parameters


def create_samples(input_parameters: InputParameters, output_directory: str, checkpoint_path: str):
    """Convenience method to generate samples."""
    noise_parameters = get_noise_parameters(input_parameters)

    sampling_parameters = PredictorCorrectorSamplingParameters(
        algorithm=input_parameters.algorithm,
        number_of_corrector_steps=input_parameters.number_of_corrector_steps,
        spatial_dimension=TOY_MODEL_PARAMETERS["spatial_dimension"],
        number_of_atoms=TOY_MODEL_PARAMETERS["number_of_atoms"],
        number_of_samples=input_parameters.number_of_samples,
        cell_dimensions=TOY_MODEL_PARAMETERS["cell_dimensions"],
        record_samples=input_parameters.record_samples,
        num_atom_types=TOY_MODEL_PARAMETERS["num_atom_types"],
        use_fixed_lattice_parameters=True
    )

    config = dict(
        noise=dataclasses.asdict(noise_parameters),
        sampling=dataclasses.asdict(sampling_parameters),
    )
    config_file = tempfile.NamedTemporaryFile(mode="w+b", delete=False)

    with open(config_file.name, "w") as fd:
        yaml.dump(config, fd)

    arguments = [
        "--config",
        config_file.name,
        "--output",
        output_directory,
        "--device",
        "cpu",
    ]

    if checkpoint_path == "analytical":
        analytical_score_network_parameters = get_analytical_score_network_parameters()
        axl_network = AnalyticalScoreNetwork(analytical_score_network_parameters)
        sys.argv.extend(arguments)
        main(axl_network=axl_network)
    else:
        arguments.extend(["--checkpoint", checkpoint_path])
        sys.argv.extend(arguments)

        main()

    os.unlink(config_file.name)


def plot_samples(output_samples_path: Path, experiment_name: str):
    """Plot samples."""
    samples = torch.load(output_samples_path / "samples.pt")
    relative_coordinates = samples[AXL_COMPOSITION].X.squeeze(-1)
    fig = plot_2d_samples(relative_coordinates)
    fig.suptitle(f"Samples drawn with {experiment_name} Score Network")
    fig.savefig(output_samples_path.parent / "score_samples.png")
    plt.close(fig)


def plot_samples_radial_distribution(output_samples_path: Path, experiment_name: str):
    """Plot samples radial distribution."""
    samples = torch.load(output_samples_path / "samples.pt")
    relative_coordinates = samples[AXL_COMPOSITION].X

    sigma_d = TOY_MODEL_PARAMETERS['sigma_d']

    list_r = torch.linspace(0, 5 * sigma_d, 1001)

    expected_distribution = list_r / sigma_d**2 * torch.exp(- 0.5 * list_r**2 / sigma_d**2)

    x0_1 = TOY_MODEL_PARAMETERS['x0_1']
    x0_2 = TOY_MODEL_PARAMETERS['x0_2']
    mu1 = torch.tensor([[[x0_1], [x0_2]]])
    mu2 = torch.tensor([[[x0_2], [x0_1]]])

    distances1 = torch.sqrt(((relative_coordinates - mu1)**2).squeeze().sum(dim=-1))
    distances2 = torch.sqrt(((relative_coordinates - mu2)**2).squeeze().sum(dim=-1))

    radial_distances = einops.rearrange([distances1, distances2], "c n -> n c").min(dim=1).values

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)

    ax = fig.add_subplot(111)

    common_params = dict(density=True, bins=100, histtype="stepfilled", alpha=0.25)
    fig.suptitle(f"Samples drawn with {experiment_name} Score Network")
    ax.plot(list_r, expected_distribution, 'g-', label='Expected distribution')
    ax.hist(radial_distances, **common_params, color='red', label=experiment_name)

    ax.set_xlim(xmin=-0.01)
    ax.legend(loc=0)
    ax.set_xlabel("Radial Distance To Closest Equilibrium Position")
    ax.set_ylabel("Count Density")
    fig.savefig(output_samples_path.parent / "samples_radial_distribution.png")
    plt.close(fig)


def get_vector_field_movie(input_parameters: InputParameters, checkpoint_path: str, output_video_path: Path):
    """Generate vector field videos."""
    analytical_score_network_parameters = get_analytical_score_network_parameters()
    analytical_score_network = AnalyticalScoreNetwork(analytical_score_network_parameters)

    if checkpoint_path == "analytical":
        axl_network = AnalyticalScoreNetwork(analytical_score_network_parameters)
    else:
        axl_network = get_axl_network(checkpoint_path)

    noise_parameters = get_noise_parameters(input_parameters)

    generate_vector_field_video(
        axl_network=axl_network,
        analytical_score_network=analytical_score_network,
        noise_parameters=noise_parameters,
        output_file_path=output_video_path,
    )


def plot_marginal_distribution(input_parameters: InputParameters, output_directory: Path):
    """Plot marginal distribution."""
    noise_parameters = get_noise_parameters(input_parameters)

    x0_1 = TOY_MODEL_PARAMETERS["x0_1"]
    x0_2 = TOY_MODEL_PARAMETERS["x0_2"]
    sigma_d = TOY_MODEL_PARAMETERS["sigma_d"]

    # Plot the marginal probability distribution.
    x = torch.linspace(0, 1, 1001)
    relative_coordinates1 = einops.rearrange(x - x0_1, "batch -> batch 1 1")
    relative_coordinates2 = einops.rearrange(x - x0_2, "batch -> batch 1 1")

    list_times = torch.linspace(0.0, 1.0, 5)
    list_sigmas = VarianceScheduler(noise_parameters).get_sigma(list_times).numpy()
    list_sigmas[0] = 0.0

    list_marginal_probabilities = []
    for time, sigma in zip(list_times, list_sigmas):

        effective_sigmas = np.sqrt(sigma_d**2 + sigma**2) * torch.ones_like(
            relative_coordinates1
        )

        log_wrapped_gaussian1 = get_log_wrapped_gaussians(
            relative_coordinates1, effective_sigmas, kmax=5
        )
        wrapped_gaussian1 = torch.exp(log_wrapped_gaussian1)
        log_wrapped_gaussian2 = get_log_wrapped_gaussians(
            relative_coordinates2, effective_sigmas, kmax=5
        )
        wrapped_gaussian2 = torch.exp(log_wrapped_gaussian2)
        marginal_probability = 0.5 * (wrapped_gaussian1 + wrapped_gaussian2)

        list_marginal_probabilities.append(marginal_probability)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(r"Marginal Probability Distribution: $\sigma_{d}$ = " + f"{sigma_d}")
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    lw = 4
    for t, sigma, marginal_probability in zip(
        list_times, list_sigmas, list_marginal_probabilities
    ):
        label = "$t$ = " + f"{t}" + r", $\sigma(t)$ =" + f"{sigma:3.2f}"
        ax.plot(x, marginal_probability, lw=lw, label=label)
        lw = 2

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(ymin=-0.01)
    ax.legend(loc=0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$P(x)$")
    plt.savefig(output_directory / "marginal_probability_distribution.png")
    plt.close(fig)

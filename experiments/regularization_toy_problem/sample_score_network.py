import dataclasses
import glob
import os
import sys
import tempfile

import yaml

from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.sample_diffusion import main
from experiments.regularization_toy_problem import EXPERIMENTS_DIR, RESULTS_DIR


def create_samples(input_parameters: dict, output_directory: str, checkpoint_path: str):
    """Convenience method to generate samples."""
    algorithm = input_parameters["algorithm"]
    total_time_steps = input_parameters["total_time_steps"]
    number_of_corrector_steps = input_parameters["number_of_corrector_steps"]
    corrector_r = input_parameters["corrector_r"]
    number_of_samples = input_parameters["number_of_samples"]
    record_samples = input_parameters["record_samples"]

    number_of_atoms = 2
    spatial_dimension = 1
    num_atom_types = 1
    cell_dimensions = [1.0]
    sigma_min = 0.001
    sigma_max = 0.2

    x0_1 = 0.25
    x0_2 = 0.75
    sigma_d = 0.01
    kmax = 5

    noise_parameters = NoiseParameters(
        total_time_steps=total_time_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        corrector_r=corrector_r,
    )

    analytical_score_network_parameters = AnalyticalScoreNetworkParameters(
        number_of_atoms=number_of_atoms,
        num_atom_types=num_atom_types,
        kmax=kmax,
        equilibrium_relative_coordinates=[[x0_1], [x0_2]],
        sigma_d=sigma_d,
        spatial_dimension=spatial_dimension,
        use_permutation_invariance=True,
    )

    sampling_parameters = PredictorCorrectorSamplingParameters(
        algorithm=algorithm,
        number_of_corrector_steps=number_of_corrector_steps,
        spatial_dimension=spatial_dimension,
        number_of_atoms=number_of_atoms,
        number_of_samples=number_of_samples,
        cell_dimensions=cell_dimensions,
        record_samples=record_samples,
        num_atom_types=num_atom_types,
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
        axl_network = AnalyticalScoreNetwork(analytical_score_network_parameters)
        sys.argv.extend(arguments)
        main(axl_network=axl_network)
    else:
        arguments.extend(["--checkpoint", checkpoint_path])
        sys.argv.extend(arguments)

        main()

    os.unlink(config_file.name)


# Parameters
model = "analytical"
# model = "no_regularizer"


algorithm = "predictor_corrector"
nt = 10
nc = 0

corrector_r = 0.4
number_of_samples = 10_000

record_samples = False


if model == "analytical":
    checkpoint_path = "analytical"
    output_directory = RESULTS_DIR / "analytical_score_network_samples"
    output_directory.mkdir(parents=True, exist_ok=True)

elif model == "no_regularizer":
    checkpoint_path = glob.glob(
        str(EXPERIMENTS_DIR / "no_regularizer/**/last_model*.ckpt"), recursive=True
    )[0]
    output_directory = RESULTS_DIR / "mlp_score_network_samples_no_regularizer"
    output_directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    input_parameters = dict(
        algorithm=algorithm,
        total_time_steps=nt,
        number_of_corrector_steps=nc,
        corrector_r=corrector_r,
        number_of_samples=number_of_samples,
        record_samples=record_samples,
    )

    create_samples(
        input_parameters,
        output_directory=str(output_directory),
        checkpoint_path=checkpoint_path,
    )

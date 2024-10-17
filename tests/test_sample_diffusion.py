import dataclasses

import pytest
import torch
import yaml

from crystal_diffusion import sample_diffusion
from crystal_diffusion.generators.predictor_corrector_position_generator import \
    PredictorCorrectorSamplingParameters
from crystal_diffusion.models.loss import MSELossParameters
from crystal_diffusion.models.optimizer import OptimizerParameters
from crystal_diffusion.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from crystal_diffusion.models.score_networks.mlp_score_network import \
    MLPScoreNetworkParameters
from crystal_diffusion.namespace import RELATIVE_COORDINATES
from crystal_diffusion.samplers.variance_sampler import NoiseParameters


@pytest.fixture()
def spatial_dimension():
    return 3


@pytest.fixture()
def number_of_atoms():
    return 8


@pytest.fixture()
def number_of_samples():
    return 12


@pytest.fixture()
def cell_dimensions():
    return [5.1, 6.2, 7.3]


@pytest.fixture(params=[True, False])
def record_samples(request):
    return request.param


@pytest.fixture()
def noise_parameters():
    return NoiseParameters(total_time_steps=10)


@pytest.fixture()
def sampling_parameters(
    number_of_atoms,
    spatial_dimension,
    number_of_samples,
    cell_dimensions,
    record_samples,
):
    return PredictorCorrectorSamplingParameters(
        number_of_corrector_steps=1,
        spatial_dimension=spatial_dimension,
        number_of_atoms=number_of_atoms,
        number_of_samples=number_of_samples,
        cell_dimensions=cell_dimensions,
        record_samples=record_samples,
    )


@pytest.fixture()
def sigma_normalized_score_network(number_of_atoms, noise_parameters):
    score_network_parameters = MLPScoreNetworkParameters(
        number_of_atoms=number_of_atoms,
        embedding_dimensions_size=8,
        n_hidden_dimensions=2,
        hidden_dimensions_size=16,
    )

    diffusion_params = PositionDiffusionParameters(
        score_network_parameters=score_network_parameters,
        loss_parameters=MSELossParameters(),
        optimizer_parameters=OptimizerParameters(name="adam", learning_rate=1e-3),
        scheduler_parameters=None,
        noise_parameters=noise_parameters,
        diffusion_sampling_parameters=None,
    )

    model = PositionDiffusionLightningModel(diffusion_params)
    return model.sigma_normalized_score_network


@pytest.fixture()
def config_path(tmp_path, noise_parameters, sampling_parameters):
    config_path = str(tmp_path / "test_config.yaml")

    config = dict(
        noise=dataclasses.asdict(noise_parameters),
        sampling=dataclasses.asdict(sampling_parameters),
    )

    with open(config_path, "w") as fd:
        yaml.dump(config, fd)

    return config_path


@pytest.fixture()
def checkpoint_path(tmp_path):
    path_to_checkpoint = tmp_path / "fake_checkpoint.pt"
    with open(path_to_checkpoint, "w") as fd:
        fd.write("This is a dummy checkpoint file.")
    return path_to_checkpoint


@pytest.fixture()
def output_path(tmp_path):
    output = tmp_path / "output"
    return output


@pytest.fixture()
def args(config_path, checkpoint_path, output_path):
    """Input arguments for main."""
    input_args = [
        f"--config={config_path}",
        f"--checkpoint={checkpoint_path}",
        f"--output={output_path}",
        "--device=cpu",
    ]

    return input_args


def test_sample_diffusion(
    mocker,
    args,
    sigma_normalized_score_network,
    output_path,
    number_of_samples,
    number_of_atoms,
    spatial_dimension,
    record_samples,
):
    mocker.patch(
        "crystal_diffusion.sample_diffusion.get_sigma_normalized_score_network",
        return_value=sigma_normalized_score_network,
    )

    sample_diffusion.main(args)

    assert (output_path / "samples.pt").exists()
    samples = torch.load(output_path / "samples.pt")
    assert samples[RELATIVE_COORDINATES].shape == (
        number_of_samples,
        number_of_atoms,
        spatial_dimension,
    )

    assert (output_path / "trajectories.pt").exists() == record_samples

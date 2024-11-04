import dataclasses

import pytest
import torch
import yaml

from diffusion_for_multi_scale_molecular_dynamics import sample_diffusion
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_position_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import \
    MSELossParameters
from diffusion_for_multi_scale_molecular_dynamics.models.axl_diffusion_lightning_model import (
    AXLDiffusionLightningModel, AXLDiffusionParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.optimizer import \
    OptimizerParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mlp_score_network import \
    MLPScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import \
    RELATIVE_COORDINATES
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters


@pytest.fixture()
def spatial_dimension():
    return 3


@pytest.fixture()
def number_of_atoms():
    return 8


@pytest.fixture()
def num_atom_types():
    return 3


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
    num_atom_types,
):
    return PredictorCorrectorSamplingParameters(
        number_of_corrector_steps=1,
        spatial_dimension=spatial_dimension,
        number_of_atoms=number_of_atoms,
        number_of_samples=number_of_samples,
        cell_dimensions=cell_dimensions,
        record_samples=record_samples,
        num_atom_types=num_atom_types,
    )


@pytest.fixture()
def sigma_normalized_score_network(number_of_atoms, noise_parameters, num_atom_types):
    score_network_parameters = MLPScoreNetworkParameters(
        number_of_atoms=number_of_atoms,
        num_atom_types=num_atom_types,
        noise_embedding_dimensions_size=8,
        atom_type_embedding_dimensions_size=8,
        n_hidden_dimensions=2,
        hidden_dimensions_size=16,
    )

    diffusion_params = AXLDiffusionParameters(
        score_network_parameters=score_network_parameters,
        loss_parameters=MSELossParameters(),
        optimizer_parameters=OptimizerParameters(name="adam", learning_rate=1e-3),
        scheduler_parameters=None,
        noise_parameters=noise_parameters,
        diffusion_sampling_parameters=None,
    )

    model = AXLDiffusionLightningModel(diffusion_params)
    return model.score_network


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
        "diffusion_for_multi_scale_molecular_dynamics.sample_diffusion.get_sigma_normalized_score_network",
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

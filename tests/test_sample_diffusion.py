import dataclasses

import pytest
import torch
import yaml

from diffusion_for_multi_scale_molecular_dynamics import sample_diffusion
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import \
    MSELossParameters
from diffusion_for_multi_scale_molecular_dynamics.models.axl_diffusion_lightning_model import (
    AXLDiffusionLightningModel, AXLDiffusionParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.optimizer import \
    OptimizerParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mlp_score_network import \
    MLPScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, AXL_COMPOSITION)
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


@pytest.fixture()
def reference_composition(num_atom_types, number_of_atoms, spatial_dimension):
    a = torch.randint(0, num_atom_types, (number_of_atoms,))
    x = torch.rand(number_of_atoms, spatial_dimension)
    lat = torch.rand(spatial_dimension, spatial_dimension)
    return AXL(A=a, X=x, L=lat)


@pytest.fixture()
def constrained_atom_indices(number_of_atoms):
    return torch.sort(torch.randperm(number_of_atoms)[:number_of_atoms // 2]).values


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
def axl_network(number_of_atoms, noise_parameters, num_atom_types, device):
    score_network_parameters = MLPScoreNetworkParameters(
        number_of_atoms=number_of_atoms,
        num_atom_types=num_atom_types,
        noise_embedding_dimensions_size=8,
        time_embedding_dimensions_size=8,
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

    model = AXLDiffusionLightningModel(diffusion_params).to(device)
    return model.axl_network


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


@pytest.fixture(params=[True, False])
def apply_constraint(request):
    return request.param


@pytest.fixture()
def constraint_data_pickle_path(tmp_path, reference_composition, constrained_atom_indices):
    path_to_pickle = tmp_path / "pickle_path.pkl"
    data = dict(reference_composition=reference_composition,
                constrained_atom_indices=constrained_atom_indices)
    torch.save(data, path_to_pickle)
    return path_to_pickle


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
def args(config_path, checkpoint_path, output_path, constraint_data_pickle_path, apply_constraint, device):
    """Input arguments for main."""
    input_args = [
        f"--config={config_path}",
        f"--checkpoint={checkpoint_path}",
        f"--output={output_path}",
        f"--device={device}",
    ]

    if apply_constraint:
        input_args.append(f"--path_to_constraint_data_pickle={constraint_data_pickle_path}")

    return input_args


def test_sample_diffusion(
    mocker,
    device,
    args,
    axl_network,
    output_path,
    number_of_samples,
    number_of_atoms,
    spatial_dimension,
    record_samples,
    apply_constraint,
    reference_composition,
    constrained_atom_indices
):
    mocker.patch(
        "diffusion_for_multi_scale_molecular_dynamics.sample_diffusion.get_axl_network",
        return_value=axl_network,
    )

    sample_diffusion.main(args)

    assert (output_path / "samples.pt").exists()
    samples = torch.load(output_path / "samples.pt")
    compositions = samples[AXL_COMPOSITION]

    assert compositions.X.shape == (
        number_of_samples,
        number_of_atoms,
        spatial_dimension,
    )
    assert compositions.A.shape == (
        number_of_samples,
        number_of_atoms,
    )

    assert (output_path / "trajectories.pt").exists() == record_samples

    if apply_constraint:
        reference_x = reference_composition.X[constrained_atom_indices].to(device)
        reference_a = reference_composition.A[constrained_atom_indices].to(device)

        assert (compositions.X[:, constrained_atom_indices] == reference_x).all()
        assert (compositions.A[:, constrained_atom_indices] == reference_a).all()

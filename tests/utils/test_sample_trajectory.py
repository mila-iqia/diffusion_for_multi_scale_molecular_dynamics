from copy import deepcopy

import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, AXL_NAME_DICT)
from diffusion_for_multi_scale_molecular_dynamics.utils.sample_trajectory import \
    PredictorCorrectorSampleTrajectory


@pytest.fixture(autouse=True, scope="module")
def set_seed():
    torch.manual_seed(32434)


@pytest.fixture(scope="module")
def number_of_predictor_steps():
    return 8


@pytest.fixture(scope="module")
def number_of_corrector_steps():
    return 3


@pytest.fixture(scope="module")
def batch_size():
    return 4


@pytest.fixture(scope="module")
def number_of_atoms():
    return 8


@pytest.fixture(scope="module")
def spatial_dimension():
    return 3


@pytest.fixture(scope="module")
def num_classes():
    return 5


@pytest.fixture(scope="module")
def basis_vectors(batch_size):
    # orthogonal boxes with dimensions between 5 and 10.
    orthogonal_boxes = torch.stack(
        [torch.diag(5.0 + 5.0 * torch.rand(3)) for _ in range(batch_size)]
    )
    # add a bit of noise to make the vectors not quite orthogonal
    basis_vectors = orthogonal_boxes + 0.1 * torch.randn(batch_size, 3, 3)
    return basis_vectors


@pytest.fixture(scope="module")
def list_i_indices(number_of_predictor_steps):
    return torch.arange(number_of_predictor_steps - 1, -1, -1)


@pytest.fixture(scope="module")
def list_sigmas(number_of_predictor_steps):
    return torch.rand(number_of_predictor_steps)


@pytest.fixture(scope="module")
def list_times(number_of_predictor_steps):
    return torch.rand(number_of_predictor_steps)


@pytest.fixture(scope="module")
def predictor_model_outputs(
    number_of_predictor_steps,
    batch_size,
    number_of_atoms,
    spatial_dimension,
    num_classes,
):
    list_scores = [
        AXL(
            A=torch.rand(batch_size, number_of_atoms, num_classes),
            X=torch.rand(batch_size, number_of_atoms, spatial_dimension),
            L=torch.zeros(
                batch_size, number_of_atoms, spatial_dimension * (spatial_dimension - 1)
            ),  # TODO placeholder
        )
        for _ in range(number_of_predictor_steps)
    ]
    return list_scores


@pytest.fixture(scope="module")
def list_x_i(number_of_predictor_steps, batch_size, number_of_atoms, spatial_dimension):
    return torch.rand(
        number_of_predictor_steps, batch_size, number_of_atoms, spatial_dimension
    )


@pytest.fixture(scope="module")
def list_x_im1(
    number_of_predictor_steps, batch_size, number_of_atoms, spatial_dimension
):
    return torch.rand(
        number_of_predictor_steps, batch_size, number_of_atoms, spatial_dimension
    )


@pytest.fixture(scope="module")
def list_atom_types_i(
    number_of_predictor_steps, batch_size, number_of_atoms, num_classes
):
    return torch.randint(
        0, num_classes, (number_of_predictor_steps, batch_size, number_of_atoms)
    )


@pytest.fixture(scope="module")
def list_atom_types_im1(
    number_of_predictor_steps, batch_size, number_of_atoms, num_classes
):
    return torch.randint(
        0, num_classes, (number_of_predictor_steps, batch_size, number_of_atoms)
    )


@pytest.fixture(scope="module")
def list_axl_i(list_x_i, list_atom_types_i):
    list_axl = [
        AXL(A=atom_types_i, X=x_i, L=torch.zeros_like(x_i))
        for atom_types_i, x_i in zip(list_atom_types_i, list_x_i)
    ]
    return list_axl


@pytest.fixture(scope="module")
def list_axl_im1(list_x_im1, list_atom_types_im1):
    list_axl = [
        AXL(A=atom_types_im1, X=x_im1, L=torch.zeros_like(x_im1))
        for atom_types_im1, x_im1 in zip(list_atom_types_im1, list_x_im1)
    ]
    return list_axl


@pytest.fixture(scope="module")
def corrector_model_outputs(
    number_of_predictor_steps,
    number_of_corrector_steps,
    batch_size,
    number_of_atoms,
    spatial_dimension,
    num_classes,
):
    list_scores = [
        AXL(
            A=torch.rand(batch_size, number_of_atoms, num_classes),
            X=torch.rand(batch_size, number_of_atoms, spatial_dimension),
            L=torch.zeros(
                batch_size, number_of_atoms, spatial_dimension * (spatial_dimension - 1)
            ),  # TODO placeholder
        )
        for _ in range(number_of_predictor_steps * number_of_corrector_steps)
    ]
    return list_scores


@pytest.fixture(scope="module")
def list_x_i_corr(
    number_of_predictor_steps,
    number_of_corrector_steps,
    batch_size,
    number_of_atoms,
    spatial_dimension,
):
    number_of_scores = number_of_predictor_steps * number_of_corrector_steps
    return torch.rand(number_of_scores, batch_size, number_of_atoms, spatial_dimension)


@pytest.fixture(scope="module")
def list_atom_types_i_corr(
    number_of_predictor_steps,
    number_of_corrector_steps,
    batch_size,
    number_of_atoms,
    num_classes,
):
    number_of_scores = number_of_predictor_steps * number_of_corrector_steps
    return torch.randint(
        0, num_classes, (number_of_scores, batch_size, number_of_atoms)
    )


@pytest.fixture(scope="module")
def list_axl_i_corr(list_x_i_corr, list_atom_types_i_corr):
    list_axl = [
        AXL(A=atom_types_i_corr, X=x_i_corr, L=torch.zeros_like(x_i_corr))
        for atom_types_i_corr, x_i_corr in zip(list_atom_types_i_corr, list_x_i_corr)
    ]
    return list_axl


@pytest.fixture(scope="module")
def list_corrected_x_i(
    number_of_predictor_steps,
    number_of_corrector_steps,
    batch_size,
    number_of_atoms,
    spatial_dimension,
):
    number_of_scores = number_of_predictor_steps * number_of_corrector_steps
    return torch.rand(number_of_scores, batch_size, number_of_atoms, spatial_dimension)


@pytest.fixture(scope="module")
def list_corrected_atom_types_i(
    number_of_predictor_steps,
    number_of_corrector_steps,
    batch_size,
    number_of_atoms,
    num_classes,
):
    number_of_scores = number_of_predictor_steps * number_of_corrector_steps
    return torch.randint(
        0, num_classes, (number_of_scores, batch_size, number_of_atoms)
    )


@pytest.fixture(scope="module")
def list_corrected_axl_i(list_corrected_x_i, list_corrected_atom_types_i):
    list_axl = [
        AXL(
            A=corrected_atom_types_i, X=corrected_x_i, L=torch.zeros_like(corrected_x_i)
        )
        for corrected_atom_types_i, corrected_x_i in zip(
            list_corrected_atom_types_i, list_corrected_x_i
        )
    ]
    return list_axl


@pytest.fixture(scope="module")
def sample_trajectory(
    number_of_corrector_steps,
    list_i_indices,
    list_times,
    list_sigmas,
    basis_vectors,
    list_axl_i,
    list_axl_im1,
    predictor_model_outputs,
    list_axl_i_corr,
    list_corrected_axl_i,
    corrector_model_outputs,
):
    sample_trajectory = PredictorCorrectorSampleTrajectory()
    sample_trajectory.record_unit_cell(basis_vectors)

    total_corrector_index = 0

    for i_index, time, sigma, axl_i, axl_im1, model_predictions_i in zip(
        list_i_indices,
        list_times,
        list_sigmas,
        list_axl_i,
        list_axl_im1,
        predictor_model_outputs,
    ):
        sample_trajectory.record_predictor_step(
            i_index=i_index,
            time=time,
            sigma=sigma,
            composition_i=axl_i,
            composition_im1=axl_im1,
            model_predictions_i=model_predictions_i,
        )

        for _ in range(number_of_corrector_steps):
            axl_i = list_axl_i_corr[total_corrector_index]
            corrected_axl_i = list_corrected_axl_i[total_corrector_index]
            model_predictions_i = corrector_model_outputs[total_corrector_index]
            sample_trajectory.record_corrector_step(
                i_index=i_index,
                time=time,
                sigma=sigma,
                composition_i=axl_i,
                corrected_composition_i=corrected_axl_i,
                model_predictions_i=model_predictions_i,
            )
            total_corrector_index += 1

    return sample_trajectory


def test_sample_trajectory_unit_cell(sample_trajectory, basis_vectors):
    torch.testing.assert_close(sample_trajectory.data["unit_cell"], basis_vectors)


def test_record_predictor(
    sample_trajectory,
    list_times,
    list_sigmas,
    list_axl_i,
    list_axl_im1,
    predictor_model_outputs,
):
    torch.testing.assert_close(
        torch.tensor(sample_trajectory.data["predictor_time"]), list_times
    )
    torch.testing.assert_close(
        torch.tensor(sample_trajectory.data["predictor_sigma"]), list_sigmas
    )
    for axl_field, axl_name in AXL_NAME_DICT.items():
        predictor_i = torch.stack(
            sample_trajectory.data[f"predictor_{axl_name}_i"], dim=0
        )
        target_predictor_i = torch.stack(
            [getattr(axl, axl_field) for axl in list_axl_i], dim=0
        )
        torch.testing.assert_close(predictor_i, target_predictor_i)
        predictor_im1 = torch.stack(
            sample_trajectory.data[f"predictor_{axl_name}_im1"], dim=0
        )
        target_predictor_im1 = torch.stack(
            [getattr(axl, axl_field) for axl in list_axl_im1], dim=0
        )
        torch.testing.assert_close(predictor_im1, target_predictor_im1)

        predictor_mo_i = torch.stack(
            sample_trajectory.data[f"predictor_{axl_name}_model_predictions"], dim=0
        )
        target_predictor_model_outputs = torch.stack(
            [getattr(axl, axl_field) for axl in predictor_model_outputs], dim=0
        )
        torch.testing.assert_close(predictor_mo_i, target_predictor_model_outputs)


def test_record_corrector(
    sample_trajectory,
    number_of_corrector_steps,
    list_times,
    list_sigmas,
    list_axl_i_corr,
    list_corrected_axl_i,
    corrector_model_outputs,
):

    torch.testing.assert_close(
        torch.tensor(sample_trajectory.data["corrector_time"]),
        torch.repeat_interleave(list_times, number_of_corrector_steps),
    )
    torch.testing.assert_close(
        torch.tensor(sample_trajectory.data["corrector_sigma"]),
        torch.repeat_interleave(list_sigmas, number_of_corrector_steps),
    )
    for axl_field, axl_name in AXL_NAME_DICT.items():
        corrector_i = torch.stack(
            sample_trajectory.data[f"corrector_{axl_name}_i"], dim=0
        )
        target_corrector_i = torch.stack(
            [getattr(axl, axl_field) for axl in list_axl_i_corr], dim=0
        )
        torch.testing.assert_close(corrector_i, target_corrector_i)
        corrector_corrected_im1 = torch.stack(
            sample_trajectory.data[f"corrector_{axl_name}_corrected_i"], dim=0
        )
        target_corrector_corrected_im1 = torch.stack(
            [getattr(axl, axl_field) for axl in list_corrected_axl_i], dim=0
        )
        torch.testing.assert_close(
            corrector_corrected_im1, target_corrector_corrected_im1
        )

        corrector_mo_i = torch.stack(
            sample_trajectory.data[f"corrector_{axl_name}_model_predictions"], dim=0
        )
        target_corrector_model_outputs = torch.stack(
            [getattr(axl, axl_field) for axl in corrector_model_outputs], dim=0
        )
        torch.testing.assert_close(corrector_mo_i, target_corrector_model_outputs)


def test_standardize_data_and_write_pickle(
    sample_trajectory,
    basis_vectors,
    list_times,
    list_sigmas,
    list_x_i,
    predictor_model_outputs,
    tmp_path,
):
    pickle_path = str(tmp_path / "test_pickle_path.pkl")
    sample_trajectory.write_to_pickle(pickle_path)

    with open(pickle_path, "rb") as fd:
        standardized_data = torch.load(fd)

    reordered_scores = einops.rearrange(
        torch.stack([axl.X for axl in predictor_model_outputs], dim=0),
        "t b n d -> b t n d",
    )
    reordered_relative_coordinates = einops.rearrange(list_x_i, "t b n d -> b t n d")

    torch.testing.assert_close(standardized_data["unit_cell"], basis_vectors)
    torch.testing.assert_close(standardized_data["time"], list_times)
    torch.testing.assert_close(standardized_data["sigma"], list_sigmas)
    torch.testing.assert_close(
        standardized_data["relative_coordinates"], reordered_relative_coordinates
    )
    torch.testing.assert_close(standardized_data["normalized_scores"], reordered_scores)


def test_reset(sample_trajectory, tmp_path):
    # We don't want to affect other tests!
    copied_sample_trajectory = deepcopy(sample_trajectory)
    assert len(copied_sample_trajectory.data.keys()) != 0
    copied_sample_trajectory.reset()
    assert len(copied_sample_trajectory.data.keys()) == 0

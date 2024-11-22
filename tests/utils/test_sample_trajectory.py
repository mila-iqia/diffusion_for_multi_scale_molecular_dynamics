from copy import deepcopy

import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.sample_trajectory import \
    SampleTrajectory


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
def list_time_indices(number_of_predictor_steps):
    return torch.arange(number_of_predictor_steps - 1, -1, -1)


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
        list_time_indices,
        basis_vectors,
        list_axl_i,
        list_axl_im1,
        predictor_model_outputs,
        list_axl_i_corr,
        list_corrected_axl_i,
        corrector_model_outputs,
):
    sample_trajectory_recorder = SampleTrajectory()
    total_corrector_index = 0

    for time_step_index, axl_i, axl_im1, model_predictions_i in zip(
        list_time_indices,
        list_axl_i,
        list_axl_im1,
        predictor_model_outputs,
    ):
        entry = dict(time_step_index=time_step_index,
                     composition_i=axl_i,
                     composition_im1=axl_im1,
                     model_predictions_i=model_predictions_i)
        sample_trajectory_recorder.record(key="predictor_step", entry=entry)

        for _ in range(number_of_corrector_steps):
            axl_i = list_axl_i_corr[total_corrector_index]
            corrected_axl_i = list_corrected_axl_i[total_corrector_index]
            model_predictions_i = corrector_model_outputs[total_corrector_index]
            entry = dict(time_step_index=time_step_index,
                         composition_i=axl_i,
                         corrected_composition_i=corrected_axl_i,
                         model_predictions_i=model_predictions_i)
            sample_trajectory_recorder.record(key="corrector_step", entry=entry)

            total_corrector_index += 1

    return sample_trajectory_recorder


@pytest.fixture(scope="module")
def pickle_data(sample_trajectory, tmp_path_factory):
    path_to_pickle = tmp_path_factory.mktemp("sample_trajectory") / "test.pkl"
    sample_trajectory.write_to_pickle(path_to_pickle)
    data = torch.load(path_to_pickle)
    return data


def test_predictor_step(number_of_predictor_steps,
                        pickle_data,
                        list_time_indices,
                        list_axl_i,
                        list_axl_im1,
                        predictor_model_outputs):
    assert "predictor_step" in pickle_data
    predictor_step_data = pickle_data["predictor_step"]

    assert len(predictor_step_data) == number_of_predictor_steps

    for step_idx in range(number_of_predictor_steps):
        entry = predictor_step_data[step_idx]
        assert entry['time_step_index'] == list_time_indices[step_idx]
        torch.testing.assert_close(entry['composition_i'], list_axl_i[step_idx])
        torch.testing.assert_close(entry['composition_im1'], list_axl_im1[step_idx])
        torch.testing.assert_close(entry['model_predictions_i'], predictor_model_outputs[step_idx])


def test_corrector_step(
    number_of_predictor_steps,
    number_of_corrector_steps,
    pickle_data,
    list_time_indices,
    list_axl_i_corr,
    list_corrected_axl_i,
    corrector_model_outputs,
):

    assert "corrector_step" in pickle_data
    corrector_step_data = pickle_data["corrector_step"]

    assert len(corrector_step_data) == number_of_predictor_steps * number_of_corrector_steps

    global_step_idx = 0
    for predictor_step_idx in range(number_of_predictor_steps):
        expected_time_index = list_time_indices[predictor_step_idx]

        for corrector_step_idx in range(number_of_corrector_steps):
            entry = corrector_step_data[global_step_idx]
            assert entry['time_step_index'] == expected_time_index
            torch.testing.assert_close(entry['composition_i'], list_axl_i_corr[global_step_idx])
            torch.testing.assert_close(entry['corrected_composition_i'], list_corrected_axl_i[global_step_idx])
            torch.testing.assert_close(entry['model_predictions_i'], corrector_model_outputs[global_step_idx])
            global_step_idx += 1


def test_reset(sample_trajectory):
    # We don't want to affect other tests!
    copied_sample_trajectory = deepcopy(sample_trajectory)
    assert len(copied_sample_trajectory._internal_data.keys()) != 0
    copied_sample_trajectory.reset()
    assert len(copied_sample_trajectory._internal_data.keys()) == 0

from copy import deepcopy

import einops
import pytest
import torch

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
def predictor_scores(
    number_of_predictor_steps, batch_size, number_of_atoms, spatial_dimension
):
    return torch.rand(
        number_of_predictor_steps, batch_size, number_of_atoms, spatial_dimension
    )


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
def corrector_scores(
    number_of_predictor_steps,
    number_of_corrector_steps,
    batch_size,
    number_of_atoms,
    spatial_dimension,
):
    number_of_scores = number_of_predictor_steps * number_of_corrector_steps
    return torch.rand(number_of_scores, batch_size, number_of_atoms, spatial_dimension)


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
def sample_trajectory(
    number_of_corrector_steps,
    list_i_indices,
    list_times,
    list_sigmas,
    basis_vectors,
    list_x_i,
    list_x_im1,
    predictor_scores,
    list_x_i_corr,
    list_corrected_x_i,
    corrector_scores,
):
    sample_trajectory = PredictorCorrectorSampleTrajectory()
    sample_trajectory.record_unit_cell(basis_vectors)

    total_corrector_index = 0

    for i_index, time, sigma, x_i, x_im1, scores in zip(
        list_i_indices, list_times, list_sigmas, list_x_i, list_x_im1, predictor_scores
    ):
        sample_trajectory.record_predictor_step(
            i_index=i_index, time=time, sigma=sigma, x_i=x_i, x_im1=x_im1, scores=scores
        )

        for _ in range(number_of_corrector_steps):
            x_i = list_x_i_corr[total_corrector_index]
            corrected_x_i = list_corrected_x_i[total_corrector_index]
            scores = corrector_scores[total_corrector_index]
            sample_trajectory.record_corrector_step(
                i_index=i_index,
                time=time,
                sigma=sigma,
                x_i=x_i,
                corrected_x_i=corrected_x_i,
                scores=scores,
            )
            total_corrector_index += 1

    return sample_trajectory


def test_sample_trajectory_unit_cell(sample_trajectory, basis_vectors):
    torch.testing.assert_close(sample_trajectory.data["unit_cell"], basis_vectors)


def test_record_predictor(
    sample_trajectory, list_times, list_sigmas, list_x_i, list_x_im1, predictor_scores
):
    torch.testing.assert_close(
        torch.tensor(sample_trajectory.data["predictor_time"]), list_times
    )
    torch.testing.assert_close(
        torch.tensor(sample_trajectory.data["predictor_sigma"]), list_sigmas
    )
    torch.testing.assert_close(
        torch.stack(sample_trajectory.data["predictor_x_i"], dim=0), list_x_i
    )
    torch.testing.assert_close(
        torch.stack(sample_trajectory.data["predictor_x_im1"], dim=0), list_x_im1
    )
    torch.testing.assert_close(
        torch.stack(sample_trajectory.data["predictor_scores"], dim=0), predictor_scores
    )


def test_record_corrector(
    sample_trajectory,
    number_of_corrector_steps,
    list_times,
    list_sigmas,
    list_x_i_corr,
    list_corrected_x_i,
    corrector_scores,
):

    torch.testing.assert_close(
        torch.tensor(sample_trajectory.data["corrector_time"]),
        torch.repeat_interleave(list_times, number_of_corrector_steps),
    )
    torch.testing.assert_close(
        torch.tensor(sample_trajectory.data["corrector_sigma"]),
        torch.repeat_interleave(list_sigmas, number_of_corrector_steps),
    )
    torch.testing.assert_close(
        torch.stack(sample_trajectory.data["corrector_x_i"], dim=0), list_x_i_corr
    )
    torch.testing.assert_close(
        torch.stack(sample_trajectory.data["corrector_corrected_x_i"], dim=0),
        list_corrected_x_i,
    )
    torch.testing.assert_close(
        torch.stack(sample_trajectory.data["corrector_scores"], dim=0), corrector_scores
    )


def test_standardize_data_and_write_pickle(
    sample_trajectory,
    basis_vectors,
    list_times,
    list_sigmas,
    list_x_i,
    predictor_scores,
    tmp_path,
):
    pickle_path = str(tmp_path / "test_pickle_path.pkl")
    sample_trajectory.write_to_pickle(pickle_path)

    with open(pickle_path, "rb") as fd:
        standardized_data = torch.load(fd)

    reordered_scores = einops.rearrange(predictor_scores, "t b n d -> b t n d")
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

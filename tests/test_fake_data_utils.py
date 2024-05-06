import pytest
import torch

from tests.fake_data_utils import find_aligning_permutation


@pytest.fixture(scope="module", autouse=True)
def set_random_seed():
    torch.manual_seed(92342342)


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def vector_dimension():
    return 5


@pytest.fixture
def noise(batch_size, vector_dimension):
    return 1e-8 * torch.rand(batch_size, vector_dimension)


@pytest.fixture
def permutation(batch_size):
    return torch.randperm(batch_size)


@pytest.fixture
def second_tensor(batch_size, vector_dimension):
    return torch.rand(batch_size, vector_dimension)


@pytest.fixture
def first_tensor(second_tensor, noise, permutation):
    return second_tensor[permutation] + noise


def test_find_aligning_permutation(first_tensor, second_tensor, permutation):
    computed_permutation = find_aligning_permutation(first_tensor, second_tensor)
    torch.testing.assert_close(permutation, computed_permutation)

import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.utils.maxvol import maxvol


@pytest.fixture
def random_matrix():
    return np.random.rand(10, 10)


def test_maxvol_smoketest(random_matrix):
    mv_indices, mv_coefficients = maxvol(random_matrix, 1.05)
    assert np.allclose(random_matrix, mv_coefficients.dot(random_matrix[mv_indices]))

import torch

from crystal_diffusion.utils.structure_utils import \
    get_orthogonal_basis_vectors


def test_get_orthogonal_basis_vectors():

    cell_dimensions = [12.34, 8.32, 7.12]
    batch_size = 16

    computed_basis_vectors = get_orthogonal_basis_vectors(batch_size, cell_dimensions)

    expected_basis_vectors = torch.zeros_like(computed_basis_vectors)

    for d, acell in enumerate(cell_dimensions):
        expected_basis_vectors[:, d, d] = acell
    torch.testing.assert_allclose(computed_basis_vectors, expected_basis_vectors)

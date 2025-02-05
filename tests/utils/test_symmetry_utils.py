import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.symmetry_utils import (
    factorial, get_all_permutation_indices)


class TestPermutationIndices:

    @pytest.fixture
    def number_of_atoms(self):
        return 4

    @pytest.fixture
    def dim(self):
        return 8

    @pytest.fixture
    def random_data(self, number_of_atoms, dim):
        return torch.rand(number_of_atoms, dim)

    @pytest.fixture
    def perm_indices_and_inverse_perm_indices(self, number_of_atoms, dim):
        perm_indices, inverse_perm_indices = get_all_permutation_indices(number_of_atoms)
        return perm_indices, inverse_perm_indices

    def test_shape(self, number_of_atoms, perm_indices_and_inverse_perm_indices):
        perm_indices, inverse_perm_indices = perm_indices_and_inverse_perm_indices
        number_of_permutations = factorial(number_of_atoms)
        expected_shape = (number_of_permutations, number_of_atoms)
        assert perm_indices.shape == expected_shape
        assert inverse_perm_indices.shape == expected_shape

    def test_unique(self, number_of_atoms, perm_indices_and_inverse_perm_indices):
        perm_indices, inverse_perm_indices = perm_indices_and_inverse_perm_indices

        perm_indices_set = set([tuple(list(indices)) for indices in perm_indices.numpy()])
        inv_perm_indices_set = set([tuple(list(indices)) for indices in inverse_perm_indices.numpy()])

        number_of_permutations = factorial(number_of_atoms)

        assert len(perm_indices_set) == number_of_permutations
        assert len(inv_perm_indices_set) == number_of_permutations

    def test_correct_range(self, number_of_atoms, perm_indices_and_inverse_perm_indices):
        perm_indices, inverse_perm_indices = perm_indices_and_inverse_perm_indices

        expected_sorted_indices = torch.arange(number_of_atoms)

        for sorted_indices in perm_indices.sort(dim=1).values:
            torch.testing.assert_close(expected_sorted_indices, sorted_indices)

        for sorted_indices in inverse_perm_indices.sort(dim=1).values:
            torch.testing.assert_close(expected_sorted_indices, sorted_indices)

    def test_inverse(self, number_of_atoms, perm_indices_and_inverse_perm_indices, random_data):
        perm_indices, inverse_perm_indices = perm_indices_and_inverse_perm_indices

        for indices, inverse_indices in zip(perm_indices, inverse_perm_indices):
            permuted_data = random_data[indices].clone()
            original_data = permuted_data[inverse_indices].clone()
            torch.testing.assert_close(original_data, random_data)

import itertools
from typing import Tuple

import torch


def factorial(n):
    """Factorial function."""
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def get_all_permutation_indices(number_of_atoms) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get all permutation indices.

    Produce all permutation indices to permute tensors which represent atoms.

    Args:
        number_of_atoms:  number of atoms to permute.

    Returns:
        perm_indices: indices for all permutations.
        inverse_perm_indices: indices for all inverse permutations.
    """
    perm_indices = torch.stack(
        [
            torch.tensor(perm)
            for perm in itertools.permutations(range(number_of_atoms))
        ]
    )

    inverse_perm_indices = perm_indices.argsort(dim=1)

    return perm_indices, inverse_perm_indices

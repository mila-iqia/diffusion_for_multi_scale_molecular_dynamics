import itertools

import torch


def get_cubic_point_group_symmetries(spatial_dimension: int = 3):
    """Get cubic point group symmetries."""
    permutations = [
        torch.diag(torch.ones(spatial_dimension))[[idx]] for idx in itertools.permutations(range(spatial_dimension))
    ]
    sign_changes = [
        torch.diag(torch.tensor(diag))
        for diag in itertools.product([-1.0, 1.0], repeat=spatial_dimension)
    ]
    symmetries = []
    for permutation in permutations:
        for sign_change in sign_changes:
            symmetries.append(permutation @ sign_change)

    return torch.stack(symmetries)

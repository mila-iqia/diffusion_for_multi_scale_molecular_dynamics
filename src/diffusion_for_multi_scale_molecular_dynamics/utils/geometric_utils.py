import itertools

import torch


def get_cubic_point_group_symmetries():
    """Get cubic point group symmetries."""
    permutations = [
        torch.diag(torch.ones(3))[[idx]] for idx in itertools.permutations([0, 1, 2])
    ]
    sign_changes = [
        torch.diag(torch.tensor(diag))
        for diag in itertools.product([-1.0, 1.0], repeat=3)
    ]
    symmetries = []
    for permutation in permutations:
        for sign_change in sign_changes:
            symmetries.append(permutation @ sign_change)

    return torch.stack(symmetries)

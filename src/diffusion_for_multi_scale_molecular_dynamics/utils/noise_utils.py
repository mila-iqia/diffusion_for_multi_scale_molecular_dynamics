import torch


def scale_sigma_by_number_of_atoms(
    sigma: torch.Tensor,
    number_of_atoms: torch.Tensor,
    spatial_dimension: int
):
    return sigma * torch.pow(number_of_atoms, 1 / spatial_dimension)
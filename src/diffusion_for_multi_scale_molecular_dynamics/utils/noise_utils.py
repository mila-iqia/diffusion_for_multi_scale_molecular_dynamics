import torch


def scale_sigma_by_number_of_atoms(
    sigma: torch.Tensor,
    number_of_atoms: torch.Tensor,
    spatial_dimension: int
) -> torch.Tensor:
    r"""Scale noise factor by the number of atoms.

    The variance of the noise distribution for cartesian coordinates depends on the size of the unit cell. If we assume
    the volume of a unit cell is proportional to the number of atoms, we can mitigate this variance by rescaling the
    factor :math:`\sigma` in the relative coordinates space by the number of atoms.

    .. math::

        \sigma(n) = \frac{\sigma}{\sqrt[d]n}

    with :math:`d`  the number of spatial dimensions.

    Args:
        sigma: unscaled noise factor :math:`\sigma` as a [batch_size, ...] tensor
        number_of_atoms: number of atoms in the unit cell as a [batch_size, ...] tensor
        spatial_dimension: number of spatial dimensions

    Returns:
        sigma_n : scaled sigma
    """
    return sigma / torch.pow(number_of_atoms, 1 / spatial_dimension)

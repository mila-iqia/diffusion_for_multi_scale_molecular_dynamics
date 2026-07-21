import torch


def get_sigma_for_relative_coordinates(
    sigma_angstrom: torch.Tensor,
    lattice_parameters: torch.Tensor,
    spatial_dimension: int,
) -> torch.Tensor:
    r"""Convert sigma from Angstrom units to relative coordinate units, per spatial direction.

    For each spatial direction :math:`d`, the sigma in relative coordinates is
    :math:`\sigma_{\text{rel},d} = \sigma_\text{cart} / L_{dd}`, where :math:`L_{dd}` is the diagonal
    lattice element for that direction. Each direction is converted independently.

    Args:
        sigma_angstrom: sigma in Angstrom units. Shape [batch_size].
        lattice_parameters: Voigt lattice parameters [L11, L22, L33, ...]. Shape [batch_size, num_lattice_params].
        spatial_dimension: number of spatial dimensions.

    Returns:
        sigma_rel: per-direction sigma in relative coordinate units. Shape [batch_size, spatial_dimension].
    """
    lattice_diagonals = lattice_parameters[:, :spatial_dimension]  # [batch_size, spatial_dimension]
    return sigma_angstrom.unsqueeze(-1) / lattice_diagonals  # [batch_size, spatial_dimension]


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

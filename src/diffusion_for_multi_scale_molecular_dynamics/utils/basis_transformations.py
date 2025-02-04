import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


def get_reciprocal_basis_vectors(basis_vectors: torch.Tensor) -> torch.Tensor:
    """Get reciprocal basis vectors.

    The basis vectors are of shape
          [-- a1 --]
      A = [-- a2 --]
          [-- a3 --]

    The reciprocal_basis vectors are of shape
           [ |   |   |  ]
      B =  [ b1  b2  b3 ]
           [ |   |   |  ]

    and A.B = I.

    Args:
        basis_vectors : Vectors that define the unit cell. Dimension [batch_size, spatial_dimension, spatial_dimension].

    Returns:
        reciprocal_basis_vectors: vectors that define the unit cell in reciprocal space.
    """
    reciprocal_basis_vectors = torch.inverse(basis_vectors)
    return reciprocal_basis_vectors


def get_positions_from_coordinates(
    relative_coordinates: torch.Tensor, basis_vectors: torch.Tensor
) -> torch.Tensor:
    """Get cartesian positions from relative coordinates.

    This method computes the positions in Euclidean space given the unitless coordinates and the basis vectors
    that define the unit cell.

    The positions are defined as p = c1 a1 + c2 a2 + c3 a3, which can be expressed as
        (p_x, p_y, p_z) = [c1, c2, c3] [  - a1 - ]
                                       [  - a2 - ]
                                       [  - a3 - ]

    Args:
        relative_coordinates : Unitless relative coordinates.
            Dimension [batch_size, number of vectors, spatial_dimension].
        basis_vectors : Vectors that define the unit cell. Dimension [batch_size, spatial_dimension, spatial_dimension].

    Returns:
        cartesian_positions: the point positions in Euclidean space, with units of Angstrom.
            Dimension [batch_size, number of vectors, spatial_dimension].
    """
    cartesian_positions = torch.matmul(relative_coordinates, basis_vectors)
    return cartesian_positions


def get_relative_coordinates_from_cartesian_positions(
    cartesian_positions: torch.Tensor, reciprocal_basis_vectors: torch.Tensor
) -> torch.Tensor:
    """Get relative coordinates from cartesian positions.

    This method computes the relative coordinates from the positions in Euclidean space and the reciprocal
    basis vectors.

    The positions are defined as p = c1 a1 + c2 a2 + c3 a3, which can be expressed as
        (p_x, p_y, p_z) = [c1, c2, c3] [  - a1 - ]
                                       [  - a2 - ]
                                       [  - a3 - ]

    The reciprocal basis vectors can be represented as
         [ |  |  | ]
     B = [ b1 b2 b3]
         [ |  |  | ]

    such that
        (p_x, p_y, p_z) . B = [c1, c2, c3].

    Args:
        cartesian_positions: positions in Euclidean space, with units of Angstrom.
            Dimension [batch_size, number of vectors, spatial_dimension].
        reciprocal_basis_vectors : Vectors that define the reciprocal unit cell.
        Dimension [batch_size, spatial_dimension, spatial_dimension].

    Returns:
        relative_coordinates : Unitless relative coordinates.
            Dimension [batch_size, number of vectors, spatial_dimension].
    """
    relative_coordinates = torch.matmul(cartesian_positions, reciprocal_basis_vectors)
    return relative_coordinates


def map_relative_coordinates_to_unit_cell(
    relative_coordinates: torch.Tensor,
) -> torch.Tensor:
    """Map relative coordinates back to unit cell.

    The function torch.remainder does not always bring back the relative coordinates in the range [0, 1).
    If the input is very small and negative, torch.remainder returns 1. This is problematic
    when using floats instead of doubles.

    This method makes sure that the positions are mapped back in the [0, 1) range and does reasonable
    things when the position is close to the edge.

    See issues:
        https://github.com/pytorch/pytorch/issues/37743
        https://github.com/pytorch/pytorch/issues/24861

    Args:
        relative_coordinates : relative coordinates, tensor of arbitrary shape.

    Returns:
        normalized_relative_coordinates: relative coordinates in the unit cell, ie, in the range [0, 1).
    """
    normalized_relative_coordinates = torch.remainder(relative_coordinates, 1.0)
    normalized_relative_coordinates[normalized_relative_coordinates == 1.0] = 0.0
    return normalized_relative_coordinates


def map_axl_composition_to_unit_cell(composition: AXL, device: torch.device) -> AXL:
    """Map relative coordinates in an AXL namedtuple back to unit cell and update the namedtuple.

    Args:
        composition: AXL namedtuple with atom types, relative coordinates and lattice as tensors of arbitrary shapes.
        device: device where to map the updated relative coordinates tensor

    Returns:
        normalized_composition: AXL namedtuple with relative coordinates in the unit cell i.e. in the range [0, 1).
    """
    normalized_relative_coordinates = map_relative_coordinates_to_unit_cell(
        composition.X
    ).to(device)
    normalized_composition = AXL(
        A=composition.A, X=normalized_relative_coordinates, L=composition.L
    )
    return normalized_composition


def map_lattice_parameters_to_unit_cell_vectors(
    lattice_parameters: torch.Tensor,
) -> torch.Tensor:
    """Map the lattice parameters in an AXL namedtuple back to vectors used to describe the lattice explicitly.

    The lattice parameters are a set of spatial dimension x (spatial dimension + 1) / 2 variables describing the length
    of the vectors and the angles.
    TODO we are currently assuming the angles to be fixed at 90 degrees.

    Args:
        lattice_parameters: lattice parameters used in AXL diffusion model i.e. the vector norms and angles.
            Dimension: [..., spatial dimension x (spatial dimension + 1) / 2]

    Returns:
        unit_cell_vectors: unit vectors. Dimension: [..., spatial dimension, spatial dimension].
    """
    last_dim_size = lattice_parameters.shape[-1]
    spatial_dimension = int((-1 + np.sqrt(1 + 8 * last_dim_size)) / 2)

    # TODO we assume a diagonal map here  - we need to revisit this when we introduce angles in the lattice box
    torch.allclose(
        lattice_parameters[..., spatial_dimension:],
        torch.zeros_like(lattice_parameters[..., spatial_dimension:]),
    )

    vector_lengths = lattice_parameters[..., :spatial_dimension]
    return torch.diag_embed(vector_lengths)


def get_number_of_lattice_parameters(spatial_dimension: int) -> int:
    """Compute the number of independent lattice parameters from the spatial dimension."""
    return int(spatial_dimension * (spatial_dimension + 1) / 2)

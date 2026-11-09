import einops
import torch

TWOPI = 2 * torch.pi


def get_geodesic_displacements(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Get geodesic displacements.

    Args:
        x1: relative coordinates, of dimensions [(batch_dimensions), spatial_dimension]
        x2: relative coordinates, of dimensions [(batch_dimensions), spatial_dimension]

    Returns:
        geodesic_displacements: the geodesic displacement on the hyper-torus along every dimension, of
            dimensions [(batch_dimensions), spatial_dimension].
    """
    delta = x2 - x1

    # use ATAN2 to find the geodesic_displacement along each coordinate.
    theta = TWOPI * delta

    geodesic_displacements = torch.atan2(torch.sin(theta), torch.cos(theta)) / TWOPI
    return geodesic_displacements


def get_squared_geodesic_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Get squared geodesic distance.

    Args:
        x1: a relative coordinates position, of dimensions [number_of_atoms, spatial_dimension]
        x2: a relative coordinates position, of dimensions [number_of_atoms, spatial_dimension]

    Returns:
        squared_geodesic_distance: the geodesic distance on the hyper-torus between the two input points.
    """
    return (get_geodesic_displacements(x1, x2)**2).sum()


def get_squared_geodesic_distance_cost_matrix(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Periodic squared Euclidean distances.

    Compute the geodesic square distances for all points in x1 and all points in x2, assuming
    that they are points on a hyper-torus. It is assumed that all inputs are in [0, 1).

    Args:
        x1 : array-like, shape (n_samples_1, n_features)
        x2 : array-like, shape (n_samples_2, n_features)

    Returns:
        squared_geodesic_distance_cost_matrix: squared distances between each points in x1 and x2,
            of dimensions [n_samples_1, n_samples_2].
    """
    n1, d = x1.shape
    n2, d_ = x2.shape
    assert d == d_, "The spatial dimensions are inconsistent. Review input."

    matrix_x1 = einops.repeat(x1, "n1 d -> n1 n2 d", n2=n2)
    matrix_x2 = einops.repeat(x2, "n2 d -> n1 n2 d", n1=n1)

    # Dimension [n1, n2, d]
    geodesic_displacement_matrix = get_geodesic_displacements(matrix_x1, matrix_x2)

    geodesic_squared_distances_matrix = (geodesic_displacement_matrix**2).sum(dim=2)
    return geodesic_squared_distances_matrix

import einops
import torch
from sklearn.neighbors import KDTree

from crystal_diffusion.models.score_networks.analytical_score_network import \
    AnalyticalScoreNetwork


def get_unit_cells(acell: float, spatial_dimension: int, number_of_samples: int) -> torch.Tensor:
    """Generate cubic unit cells."""
    unit_cell = torch.diag(acell * torch.ones(spatial_dimension))
    return unit_cell.repeat(number_of_samples, 1, 1)


def get_random_equilibrium_relative_coordinates(number_of_atoms: int, spatial_dimension: int) -> torch.Tensor:
    """Random equilibrium coordinates inside the unit cell."""
    # Make sure the positions are far from the edges.
    return 0.6 * torch.rand(number_of_atoms, spatial_dimension) + 0.2


def get_random_inverse_covariance(spring_constant_scale: float, number_of_atoms: int,
                                  spatial_dimension: int) -> torch.Tensor:
    """Get a random inverse covariance."""
    flat_dim = number_of_atoms * spatial_dimension

    diagonal_values = (torch.rand(flat_dim) + 0.5) * spring_constant_scale

    random_matrix = torch.rand(flat_dim, flat_dim)
    orthogonal_matrix, _, _ = torch.svd(random_matrix)

    flat_inverse_covariance = orthogonal_matrix @ (torch.diag(diagonal_values) @ orthogonal_matrix.T)

    inverse_covariance = einops.rearrange(flat_inverse_covariance, "(n1 d1) (n2 d2) -> n1 d1 n2 d2",
                                          n1=number_of_atoms, n2=number_of_atoms,
                                          d1=spatial_dimension, d2=spatial_dimension)

    return inverse_covariance


def get_exact_samples(equilibrium_relative_coordinates: torch.Tensor, inverse_covariance: torch.Tensor,
                      number_of_samples: int) -> torch.Tensor:
    """Sample the exact harmonic energy."""
    device = equilibrium_relative_coordinates.device
    natom, spatial_dimension, _, _ = inverse_covariance.shape

    flat_dim = natom * spatial_dimension
    flat_equilibrium_relative_coordinates = einops.rearrange(equilibrium_relative_coordinates, "n d -> (n d)")

    matrix = einops.rearrange(inverse_covariance, "n1 d1 n2 d2 -> (n1 d1) (n2 d2)")

    eigen = torch.linalg.eigh(matrix)
    eigenvalues = eigen.eigenvalues
    eigenvectors_as_columns = eigen.eigenvectors

    sigmas = 1. / torch.sqrt(eigenvalues)

    z_scores = torch.randn(number_of_samples, flat_dim).to(device)

    sigma_z_scores = z_scores * sigmas.unsqueeze(0)

    flat_displacements = sigma_z_scores @ eigenvectors_as_columns.T

    flat_relative_coordinates = flat_equilibrium_relative_coordinates.unsqueeze(0) + flat_displacements

    relative_coordinates_sample = einops.rearrange(flat_relative_coordinates,
                                                   "batch (n d) -> batch n d", n=natom, d=spatial_dimension)

    return relative_coordinates_sample


def get_samples_harmonic_energy(equilibrium_relative_coordinates: torch.Tensor, inverse_covariance: torch.Tensor,
                                samples: torch.Tensor) -> torch.Tensor:
    """Get samples harmonic energy."""
    all_permutations = AnalyticalScoreNetwork._get_all_equilibrium_permutations(equilibrium_relative_coordinates)

    flat_permutations = einops.rearrange(all_permutations, "perm n d -> perm (n d)")
    flat_samples = einops.rearrange(samples, "batch n d -> batch (n d)")

    # find the smallest displacements by matching the 'best' permutation to each sample
    tree = KDTree(flat_permutations.numpy())

    _, min_indices = tree.query(flat_samples, k=1)
    min_indices = min_indices[:, 0]

    flat_displacements = flat_samples - flat_permutations[min_indices]

    flat_inverse_covariance = einops.rearrange(inverse_covariance, "n1 d1 n2 d2 -> (n1 d1) (n2 d2)")

    m_u = einops.einsum(flat_inverse_covariance, flat_displacements, "flat1 flat2, batch flat2 -> batch flat1")

    u_m_u = einops.einsum(flat_displacements, m_u, "batch flat, batch flat-> batch")
    energies = 0.5 * u_m_u
    return energies


def get_relative_harmonic_energy(batch_relative_coordinates: torch.Tensor,
                                 equilibrium_relative_coordinates: torch.Tensor,
                                 spring_constant: float):
    """Get relative harmonic energy.

    This is the harmonic energy without the center of mass term when there are only two atoms
    and an isotropic dynamical matrix described by a spring constant.
    """
    assert batch_relative_coordinates.shape[1] == 2, "This method is specialized to 2 atoms only."
    assert equilibrium_relative_coordinates.shape[0] == 2, "This method is specialized to 2 atoms only."

    batch_displacements = batch_relative_coordinates[:, 1, :] - batch_relative_coordinates[:, 0, :]
    energies = spring_constant * (batch_displacements ** 2).sum(dim=1)

    return energies

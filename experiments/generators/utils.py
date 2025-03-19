from typing import Any, AnyStr, Dict

import einops
import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from diffusion_for_multi_scale_molecular_dynamics.utils.sample_trajectory import \
    SampleTrajectory


class DisplacementCalculator:
    """Calculate the displacement distribution."""

    def __init__(self, equilibrium_relative_coordinates: torch.Tensor):
        """Init method."""
        self.equilibrium_relative_coordinates = equilibrium_relative_coordinates

    def compute_displacements(
        self, batch_relative_coordinates: torch.Tensor

    ) -> np.ndarray:
        """Compute displacements."""
        return (batch_relative_coordinates - self.equilibrium_relative_coordinates).flatten().numpy()


def generate_exact_samples(
    equilibrium_relative_coordinates: torch.Tensor,
    sigma_d: float,
    number_of_samples: int,
):
    """Generate Gaussian samples about the equilibrium relative coordinates."""
    variance_parameter = sigma_d**2

    number_of_atoms, spatial_dimension = equilibrium_relative_coordinates.shape
    nd = number_of_atoms * spatial_dimension

    inverse_covariance = torch.diag(torch.ones(nd)) / variance_parameter
    inverse_covariance = inverse_covariance.reshape(
        number_of_atoms, spatial_dimension, number_of_atoms, spatial_dimension
    )

    exact_samples = get_exact_samples_harmonic_energy(
        equilibrium_relative_coordinates, inverse_covariance, number_of_samples
    )
    exact_samples = map_relative_coordinates_to_unit_cell(exact_samples)
    return exact_samples


def get_exact_samples_harmonic_energy(
    equilibrium_relative_coordinates: torch.Tensor,
    inverse_covariance: torch.Tensor,
    number_of_samples: int,
) -> torch.Tensor:
    """Get exact sample harmonic energy.

    Generate samples from a Gaussian distribution with a given inverse covariance matrix.
    This is like a sample from a Boltzmann distribution with a harmonic energy (ie, quadratic in displacements).
    """
    device = equilibrium_relative_coordinates.device
    natom, spatial_dimension, _, _ = inverse_covariance.shape

    flat_dim = natom * spatial_dimension
    flat_equilibrium_relative_coordinates = einops.rearrange(
        equilibrium_relative_coordinates, "n d -> (n d)"
    )

    matrix = einops.rearrange(inverse_covariance, "n1 d1 n2 d2 -> (n1 d1) (n2 d2)")

    eigen = torch.linalg.eigh(matrix)
    eigenvalues = eigen.eigenvalues
    eigenvectors_as_columns = eigen.eigenvectors

    sigmas = 1.0 / torch.sqrt(eigenvalues)

    z_scores = torch.randn(number_of_samples, flat_dim).to(device)

    sigma_z_scores = z_scores * sigmas.unsqueeze(0)

    flat_displacements = sigma_z_scores @ eigenvectors_as_columns.T

    flat_relative_coordinates = (
        flat_equilibrium_relative_coordinates.unsqueeze(0) + flat_displacements
    )

    relative_coordinates_sample = einops.rearrange(
        flat_relative_coordinates,
        "batch (n d) -> batch n d",
        n=natom,
        d=spatial_dimension,
    )

    return relative_coordinates_sample


def standardize_sde_trajectory_data(sde_sample_trajectory: SampleTrajectory) -> Dict[AnyStr, Any]:
    """Utility method to extract relevant data from the internal data of the SampleTrajectory object."""
    raw_data = sde_sample_trajectory._internal_data['sde'][0]

    times = raw_data['times']
    relative_coordinates = raw_data['relative_coordinates']

    return dict(time=times, relative_coordinates=relative_coordinates)

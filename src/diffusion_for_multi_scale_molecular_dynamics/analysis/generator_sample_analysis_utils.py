import torch
from einops import einops

from diffusion_for_multi_scale_molecular_dynamics.generators.ode_position_generator import (
    ExplodingVarianceODEPositionGenerator, ODESamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.graph_utils import \
    get_adj_matrix
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.variance_sampler import \
    NoiseParameters


class PartialODEPositionGenerator(ExplodingVarianceODEPositionGenerator):
    """Partial ODE Position Generator.

    This class is for exploring the nature of the sampling process. It differs from the base class
    by:
        1- providing a final time, 'tf', from which diffusion ODE solving starts (tf=1 for the actual problem).
        2- providing a fixed starting point, initial_relative_coordinates, instead of a random starting point.
    """

    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: ODESamplingParameters,
        sigma_normalized_score_network: ScoreNetwork,
        initial_relative_coordinates: torch.Tensor,
        tf: float = 1.0,
    ):
        """Init method."""
        super(PartialODEPositionGenerator, self).__init__(
            noise_parameters, sampling_parameters, sigma_normalized_score_network
        )

        self.tf = tf
        assert initial_relative_coordinates.shape[1:] == (
            sampling_parameters.number_of_atoms,
            sampling_parameters.spatial_dimension,
        ), "Inconsistent shape"

        self.initial_relative_coordinates = initial_relative_coordinates

    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        assert (
            number_of_samples == self.initial_relative_coordinates.shape[0]
        ), "Inconsistent number of samples"
        return self.initial_relative_coordinates


def get_interatomic_distances(
    cartesian_positions: torch.Tensor,
    basis_vectors: torch.Tensor,
    radial_cutoff: float = 5.0,
):
    """Get Interatomic Distances.

    Args:
        cartesian_positions :
        basis_vectors : basis vectors of defining the unit cell.
        radial_cutoff : neighbors are considered  up to this cutoff.

    Returns:
        distances : all distances up to cutoff.
    """
    shifted_adjacency_matrix, shifts, _, _ = get_adj_matrix(
        positions=cartesian_positions,
        basis_vectors=basis_vectors,
        radial_cutoff=radial_cutoff,
    )

    flat_positions = einops.rearrange(cartesian_positions, "b n d -> (b n) d")

    displacements = (
        flat_positions[shifted_adjacency_matrix[1]]
        - flat_positions[shifted_adjacency_matrix[0]]
        + shifts
    )
    interatomic_distances = torch.linalg.norm(displacements, dim=1)
    return interatomic_distances

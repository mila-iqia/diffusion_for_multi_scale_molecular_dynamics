import sys

import torch

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import \
    SamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.trajectory_initializer import \
    TrajectoryInitializer
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters

sys.path.append(str(TOP_DIR / "experiments/atom_types_only_experiments/patches")) # noqa

from fixed_starting_point_trajectory_initializer import \
    FixedStartingPointTrajectoryInitializer  # noqa


class IdentityRelativeCoordinatesUpdateLangevinGenerator(LangevinGenerator):
    """Identity Relative Coordinates Update Langevin Generator."""

    def _relative_coordinates_update(
        self,
        relative_coordinates: torch.Tensor,
        sigma_normalized_scores: torch.Tensor,
        sigma_i: torch.Tensor,
        score_weight: torch.Tensor,
        gaussian_noise_weight: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """NO-OP Relative coordinates update."""
        return relative_coordinates


def instantiate_identity_relative_coordinates_generator(
    sampling_parameters: SamplingParameters,
    noise_parameters: NoiseParameters,
    axl_network: ScoreNetwork,
    trajectory_initializer: TrajectoryInitializer,
):
    """Instantiate generator."""
    fixed_starting_point_trajectory_initializer = (
        FixedStartingPointTrajectoryInitializer()
    )

    generator = IdentityRelativeCoordinatesUpdateLangevinGenerator(
        sampling_parameters=sampling_parameters,
        noise_parameters=noise_parameters,
        axl_network=axl_network,
        trajectory_initializer=fixed_starting_point_trajectory_initializer,
    )
    return generator

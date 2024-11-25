import logging

import einops
import torch
from equilibrium_structure import create_equilibrium_sige_structure

from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters

logger = logging.getLogger(__name__)


class IdentityRelativeCoordinatesUpdateLangevinGenerator(LangevinGenerator):
    """Identity Relative Coordinates Update Langevin Generator."""
    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: PredictorCorrectorSamplingParameters,
        axl_network: ScoreNetwork,
    ):
        """Init method."""
        super().__init__(noise_parameters, sampling_parameters, axl_network)

        structure = create_equilibrium_sige_structure()
        self.fixed_relative_coordinates = torch.from_numpy(structure.frac_coords).to(
            torch.float
        )

    def initialize(
        self, number_of_samples: int, device: torch.device = torch.device("cpu")
    ):
        """Initialize method."""
        logger.debug("Initialize with fixed relative coordinates.")
        init_composition = super().initialize(number_of_samples, device=device)

        fixed_x = einops.repeat(
            self.fixed_relative_coordinates,
            "natoms space -> nsamples natoms space",
            nsamples=number_of_samples,
        ).to(init_composition.X)

        fixed_init_composition = AXL(
            A=init_composition.A, X=fixed_x, L=init_composition.L
        )

        return fixed_init_composition

    def relative_coordinates_update(
        self,
        relative_coordinates: torch.Tensor,
        sigma_normalized_scores: torch.Tensor,
        sigma_i: torch.Tensor,
        score_weight: torch.Tensor,
        gaussian_noise_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Relative coordinates update."""
        return relative_coordinates

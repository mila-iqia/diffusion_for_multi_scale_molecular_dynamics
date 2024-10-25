from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_position_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.variance_sampler import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noisers.relative_coordinates_noiser import \
    RelativeCoordinatesNoiser
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell


@dataclass(kw_only=True)
class ConstrainedLangevinGeneratorParameters(PredictorCorrectorSamplingParameters):
    """Hyper-parameters for diffusion sampling with the predictor-corrector algorithm."""

    algorithm: str = "constrained_langevin"
    constrained_relative_coordinates: (
        np.ndarray
    )  # the positions that must be satisfied at the end of sampling.


class ConstrainedLangevinGenerator(LangevinGenerator):
    """Constrained Annealed Langevin Dynamics Generator.

    This generator implements a basic version of the inpainting algorithm presented in the
    paper
        "RePaint: Inpainting using Denoising Diffusion Probabilistic Models".
    """

    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: ConstrainedLangevinGeneratorParameters,
        sigma_normalized_score_network: ScoreNetwork,
    ):
        """Init method."""
        super().__init__(
            noise_parameters, sampling_parameters, sigma_normalized_score_network
        )

        self.constraint_relative_coordinates = torch.from_numpy(
            sampling_parameters.constrained_relative_coordinates
        )

        assert (
            len(self.constraint_relative_coordinates.shape) == 2
        ), "The constrained relative coordinates have the wrong shape"

        number_of_constraints, spatial_dimensions = (
            self.constraint_relative_coordinates.shape
        )
        assert (
            number_of_constraints <= self.number_of_atoms
        ), "There are more constrained positions than atoms!"
        assert (
            spatial_dimensions <= self.spatial_dimension
        ), "The spatial dimension of the constrained positions is inconsistent"

        # Without loss of generality, we impose that the first positions are constrained.
        # This should have no consequence for a permutation equivariant model.
        self.constraint_mask = torch.zeros(self.number_of_atoms, dtype=bool)
        self.constraint_mask[:number_of_constraints] = True

        self.relative_coordinates_noiser = RelativeCoordinatesNoiser()

    def _apply_constraint(self, x: torch.Tensor, device: torch.device) -> None:
        """This method applies the coordinate constraint in place on the input configuration."""
        x[:, self.constraint_mask] = self.constraint_relative_coordinates.to(device)

    def sample(
        self, number_of_samples: int, device: torch.device, unit_cell: torch.Tensor
    ) -> torch.Tensor:
        """Sample.

        This method draws  samples, imposing the satisfaction of positional constraints.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.
            unit_cell: unit cell definition in Angstrom.
                Tensor of dimensions [number_of_samples, spatial_dimension, spatial_dimension]

        Returns:
            samples: relative coordinates samples.
        """
        assert unit_cell.size() == (
            number_of_samples,
            self.spatial_dimension,
            self.spatial_dimension,
        ), (
            "Unit cell passed to sample should be of size (number of sample, spatial dimension, spatial dimension"
            + f"Got {unit_cell.size()}"
        )

        # Initialize a configuration that satisfy the constraint, but is otherwise random.
        # Since the noising process is 'atom-per-atom', the non-constrained position should have no impact.
        x0_known = map_relative_coordinates_to_unit_cell(
            self.initialize(number_of_samples)
        ).to(device)
        self._apply_constraint(x0_known, device)

        x_ip1 = map_relative_coordinates_to_unit_cell(
            self.initialize(number_of_samples)
        ).to(device)
        forces = torch.zeros_like(x_ip1)

        broadcasting = torch.ones(
            number_of_samples, self.number_of_atoms, self.spatial_dimension
        ).to(device)

        for i in tqdm(range(self.number_of_discretization_steps - 1, -1, -1)):
            sigma_i = self.noise.sigma[i]
            broadcast_sigmas_i = sigma_i * broadcasting
            # Noise an example satisfying the constraints from t_0 to t_i
            x_i_known = self.relative_coordinates_noiser.get_noisy_relative_coordinates_sample(
                x0_known, broadcast_sigmas_i
            )
            # Denoise from t_{i+1} to t_i
            x_i = map_relative_coordinates_to_unit_cell(
                self.predictor_step(x_ip1, i + 1, unit_cell, forces)
            )

            # Combine the known and unknown
            x_i[:, self.constraint_mask] = x_i_known[:, self.constraint_mask]

            for _ in range(self.number_of_corrector_steps):
                x_i = map_relative_coordinates_to_unit_cell(
                    self.corrector_step(x_i, i, unit_cell, forces)
                )
            x_ip1 = x_i

        # apply the constraint one last time
        self._apply_constraint(x_i, device)

        return x_i

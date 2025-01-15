from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noisers.relative_coordinates_noiser import \
    RelativeCoordinatesNoiser


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
        axl_network: ScoreNetwork,
    ):
        """Init method."""
        super().__init__(noise_parameters, sampling_parameters, axl_network)

        self.constraint_relative_coordinates = torch.from_numpy(
            sampling_parameters.constrained_relative_coordinates
        )  # TODO constraint the atom type as well

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

    def _apply_constraint(self, composition: AXL, device: torch.device) -> AXL:
        """This method applies the coordinate constraint on the input configuration."""
        x = composition.X
        x[:, self.constraint_mask] = self.constraint_relative_coordinates.to(device)
        updated_axl = AXL(
            A=composition.A,
            X=x,
            L=composition.L,
        )
        return updated_axl

    def sample(
        self, number_of_samples: int, device: torch.device,
    ) -> AXL:
        """Sample.

        This method draws  samples, imposing the satisfaction of positional constraints.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.

        Returns:
            samples: composition samples as AXL namedtuple (atom types, reduced coordinates, lattice vectors)
        """
        # Initialize a configuration that satisfy the constraint, but is otherwise random.
        # Since the noising process is 'atom-per-atom', the non-constrained position should have no impact.
        composition0_known = self.initialize(number_of_samples, device)
        # this is an AXL objet

        composition0_known = self._apply_constraint(composition0_known, device)

        composition_ip1 = self.initialize(number_of_samples, device)
        forces = torch.zeros_like(composition_ip1.X)

        coordinates_broadcasting = torch.ones(
            number_of_samples, self.number_of_atoms, self.spatial_dimension
        ).to(device)

        for i in tqdm(range(self.number_of_discretization_steps - 1, -1, -1)):
            sigma_i = self.noise.sigma[i]
            broadcast_sigmas_i = sigma_i * coordinates_broadcasting
            # Noise an example satisfying the constraints from t_0 to t_i
            x_i_known = (
                self.relative_coordinates_noiser.get_noisy_relative_coordinates_sample(
                    composition0_known.X, broadcast_sigmas_i
                )
            )
            # Denoise from t_{i+1} to t_i
            composition_i = self.predictor_step(
                composition_ip1, i + 1, forces
            )

            # Combine the known and unknown
            x_i = composition_i.X
            x_i[:, self.constraint_mask] = x_i_known[:, self.constraint_mask]
            composition_i = AXL(A=composition_i.A, X=x_i, L=composition_i.L)

            for _ in range(self.number_of_corrector_steps):
                composition_i = self.corrector_step(composition_i, i, forces)

            composition_ip1 = composition_i

        # apply the constraint one last time
        composition_i = self._apply_constraint(composition_i, device)

        return composition_i

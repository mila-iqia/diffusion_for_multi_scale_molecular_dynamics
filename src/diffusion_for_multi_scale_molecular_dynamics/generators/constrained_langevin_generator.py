from dataclasses import dataclass
from typing import Optional

import torch

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.noising_transform import \
    NoisingTransform
from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.sampling_constraint import \
    SamplingConstraint
from diffusion_for_multi_scale_molecular_dynamics.generators.trajectory_initializer import \
    TrajectoryInitializer
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, AXL, LATTICE_PARAMETERS, NOISY_ATOM_TYPES,
    NOISY_RELATIVE_COORDINATES, RELATIVE_COORDINATES)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters


@dataclass(kw_only=True)
class ConstrainedLangevinGeneratorParameters(PredictorCorrectorSamplingParameters):
    """Hyper-parameters for diffusion sampling with the predictor-corrector algorithm."""
    algorithm: str = "constrained_langevin"


class ConstrainedLangevinGenerator(LangevinGenerator):
    """Constrained Annealed Langevin Dynamics Generator.

    This generator implements a basic version of the inpainting algorithm presented in the
    paper
        "RePaint: Inpainting using Denoising Diffusion Probabilistic Models".
    """
    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: PredictorCorrectorSamplingParameters,
        axl_network: ScoreNetwork,
        sampling_constraints: SamplingConstraint,
        trajectory_initializer: Optional[TrajectoryInitializer] = None,
    ):
        """Init method."""
        super().__init__(noise_parameters=noise_parameters,
                         sampling_parameters=sampling_parameters,
                         axl_network=axl_network,
                         trajectory_initializer=trajectory_initializer)

        self.sampling_constraints = sampling_constraints

        number_of_constraints, spatial_dimension = (
            self.sampling_constraints.constrained_relative_coordinates.shape
        )

        assert len(sampling_constraints.elements) == sampling_parameters.num_atom_types, \
            "Inconsistent number of atom types vs. elements list"

        assert (
            number_of_constraints <= self.number_of_atoms
        ), "There are more constrained positions than atoms!"
        assert (
            spatial_dimension <= self.spatial_dimension
        ), "The spatial dimension of the constrained relative coordinates is inconsistent"

        if self.sampling_constraints.constrained_indices is None:
            # We impose that the first positions are constrained.
            # This should have no consequence for a permutation equivariant model.
            self.constraint_indices = torch.arange(number_of_constraints)
        else:
            self.constraint_indices = self.sampling_constraints.constrained_indices

        self.noising_transform = NoisingTransform(noise_parameters=noise_parameters,
                                                  num_atom_types=sampling_parameters.num_atom_types,
                                                  spatial_dimension=sampling_parameters.spatial_dimension,
                                                  use_fixed_lattice_parameters=True,
                                                  use_optimal_transport=False)

    def _apply_constraint(self, composition: AXL, device: torch.device) -> AXL:
        """This method applies the coordinate constraint on the input configuration."""
        x = composition.X
        a = composition.A
        x[:, self.constraint_indices] = self.sampling_constraints.constrained_relative_coordinates.to(device)
        a[:, self.constraint_indices] = self.sampling_constraints.constrained_atom_types.to(device)

        updated_axl = AXL(A=a, X=x, L=composition.L)
        return updated_axl

    def _get_composition_0_known(self, number_of_samples: int, device: torch.device) -> AXL:
        """Get composition0_known.

        Initialize a configuration that satisfy the constraint, but is otherwise random.
        Since the noising process is 'atom-per-atom', the non-constrained terms should have no impact.
        """
        composition0_known = self.initialize(number_of_samples, device)
        composition0_known = self._apply_constraint(composition0_known, device)
        return composition0_known

    def predictor_step(
        self,
        composition_i: AXL,
        index_i: int,
        cartesian_forces: torch.Tensor,
    ) -> AXL:
        """Predictor step.

        We overload the base class predictor_step to apply REPAINT.

        Args:
            composition_i : sampled composition (atom types, relative coordinates, lattice vectors), at time step i.
            index_i : index of the time step.
            cartesian_forces: forces conditioning the sampling process

        Returns:
            composition_im1 : sampled composition, at time step i - 1.
        """
        raw_composition_im1 = super().predictor_step(composition_i=composition_i,
                                                     index_i=index_i,
                                                     cartesian_forces=cartesian_forces)

        composition_im1 = self._repaint_composition(raw_composition_i=raw_composition_im1,
                                                    index_i=index_i - 1)
        return composition_im1

    def _noise_composition(self, input_composition: AXL, index_i: int) -> AXL:
        """This method applies noise to the input composition."""
        if index_i == 0:
            # This must be the final denoising step, and the composition is already at t=0. Do not noise!
            return input_composition

        input_batch = {ATOM_TYPES: input_composition.A,
                       RELATIVE_COORDINATES: input_composition.X,
                       LATTICE_PARAMETERS: input_composition.L}

        output_batch = self.noising_transform.transform_given_time_index(input_batch, index_i)
        noised_composition_i = AXL(A=output_batch[NOISY_ATOM_TYPES],
                                   X=output_batch[NOISY_RELATIVE_COORDINATES],
                                   L=input_composition.L)
        return noised_composition_i

    def _repaint_composition(self, raw_composition_i: AXL, index_i: int) -> AXL:
        """Repaint composition.

        Args:
            raw_composition_i: an AXL composition for index i
            index_i: the diffusion time index i.

        Returns:
            repainted_composition_i: an AXL composition for index i, with applied constraints.
        """
        x_i = raw_composition_i.X
        a_i = raw_composition_i.A

        device = x_i.device
        number_of_samples = x_i.shape[0]

        composition_0_known = self._get_composition_0_known(number_of_samples, device)

        # Noise a composition satisfying the constraints from t_0 to t_i
        composition_i_known = self._noise_composition(input_composition=composition_0_known,
                                                      index_i=index_i)

        # Combine the known and unknown
        x_i[:, self.constraint_indices] = composition_i_known.X[:, self.constraint_indices]
        a_i[:, self.constraint_indices] = composition_i_known.A[:, self.constraint_indices]

        composition_im1 = AXL(A=a_i, X=x_i, L=raw_composition_i.L)
        return composition_im1

    def sample(
        self, number_of_samples: int, device: torch.device,
    ) -> AXL:
        """Sample.

        This method draws samples, imposing the satisfaction of positional constraints.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.

        Returns:
            samples: composition samples as AXL namedtuple (atom types, reduced coordinates, lattice vectors)
        """
        composition_i = super().sample(number_of_samples=number_of_samples, device=device)
        # apply the constraint one last time
        composition_i = self._apply_constraint(composition_i, device)
        return composition_i

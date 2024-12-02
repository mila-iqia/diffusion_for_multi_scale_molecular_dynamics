import einops
import torch
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noisers.atom_types_noiser import \
    AtomTypesNoiser
from diffusion_for_multi_scale_molecular_dynamics.noisers.relative_coordinates_noiser import \
    RelativeCoordinatesNoiser
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    class_index_to_onehot


class ConstrainedPredictorCorrectorAXLGenerator:
    """Constrained Predictor Corrector AXL Generator.

    This class constrains the input PC generator following a basic version of the inpainting algorithm
    presented in the paper
        "RePaint: Inpainting using Denoising Diffusion Probabilistic Models".
    """

    def __init__(
        self,
        generator: LangevinGenerator,
        reference_composition: AXL,
        constrained_atom_indices: torch.Tensor,
    ):
        """Init method."""
        self.generator = generator

        self.number_of_atoms = self.generator.number_of_atoms
        self.num_classes = self.generator.num_classes

        self.reference_composition = reference_composition
        self.constraint_indices = constrained_atom_indices

        assert (
            len(self.reference_composition.X.shape) == 2
        ), "The constrained relative coordinates have the wrong shape"

        assert (
            len(self.reference_composition.A.shape) == 1
        ), "The constrained atom types have the wrong shape"

        assert (
            len(constrained_atom_indices.shape) == 1
        ), "The constrained_atom_indices array has the wrong shape"

        self.relative_coordinates_noiser = RelativeCoordinatesNoiser()
        self.atom_type_noiser = AtomTypesNoiser()

    def _apply_constraint(self, composition: AXL, device: torch.device) -> AXL:
        """This method applies the coordinate constraint on the input configuration."""
        constrained_x = composition.X.clone()
        constrained_x[:, self.constraint_indices] = self.reference_composition.X[
            self.constraint_indices
        ].to(device)

        constrained_a = composition.A.clone()
        constrained_a[:, self.constraint_indices] = self.reference_composition.A[
            self.constraint_indices
        ].to(device)

        constrained_composition = AXL(
            A=constrained_a,
            X=constrained_x,
            L=composition.L,
        )
        return constrained_composition

    def _get_noised_known_composition(
        self, i: int, number_of_samples: int, device: torch.device
    ) -> AXL:
        """This method applies the noise to the known composition."""
        # Initialize compositions that satisfies the constraint, but is otherwise random.
        # Since the noising process is 'atom-per-atom', the non-constrained position should have no impact.
        composition0_known = self.generator.initialize(number_of_samples, device)
        composition0_known = self._apply_constraint(composition0_known, device)

        q_bar_matrices_i = einops.repeat(
            self.generator.noise.q_bar_matrix[i].to(device),
            "n1 n2 -> nsamples natoms n1 n2",
            nsamples=number_of_samples,
            natoms=self.number_of_atoms,
        )

        sigma_i = self.generator.noise.sigma[i]
        coordinates_broadcasting = torch.ones_like(composition0_known.X)
        broadcast_sigmas_i = sigma_i * coordinates_broadcasting

        # Noise an example satisfying the constraints from t_0 to t_i
        x_i_known = (
            self.relative_coordinates_noiser.get_noisy_relative_coordinates_sample(
                composition0_known.X, broadcast_sigmas_i
            )
        )

        one_hot_a_i = class_index_to_onehot(
            composition0_known.A, num_classes=self.num_classes
        )
        a_i_known = self.atom_type_noiser.get_noisy_atom_types_sample(
            one_hot_a_i, q_bar_matrices_i
        )

        noised_composition = AXL(A=a_i_known, X=x_i_known, L=composition0_known.L)
        return noised_composition

    def _combine_noised_and_denoised_compositions(
        self, noised_composition: AXL, denoised_composition: AXL
    ) -> AXL:

        updated_x = denoised_composition.X.clone()
        updated_a = denoised_composition.A.clone()

        updated_x[:, self.constraint_indices] = noised_composition.X[
            :, self.constraint_indices
        ]
        updated_a[:, self.constraint_indices] = noised_composition.A[
            :, self.constraint_indices
        ]

        composition_i = AXL(A=updated_a, X=updated_x, L=denoised_composition.L)
        return composition_i

    def sample(
        self, number_of_samples: int, device: torch.device, unit_cell: torch.Tensor
    ) -> AXL:
        """Sample.

        This method draws samples, imposing the satisfaction of positional constraints.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.
            unit_cell: unit cell definition in Angstrom.
                Tensor of dimensions [number_of_samples, spatial_dimension, spatial_dimension]

        Returns:
            samples: composition samples as AXL namedtuple (atom types, reduced coordinates, lattice vectors)
        """
        assert unit_cell.size() == (
            number_of_samples,
            self.generator.spatial_dimension,
            self.generator.spatial_dimension,
        ), (
            "Unit cell passed to sample should be of size (number of sample, spatial dimension, spatial dimension"
            + f"Got {unit_cell.size()}"
        )

        composition_ip1 = self.generator.initialize(number_of_samples, device)
        forces = torch.zeros_like(composition_ip1.X)

        for i in tqdm(range(self.generator.number_of_discretization_steps - 1, -1, -1)):

            # Noise from t_0 to t_i
            noised_composition_i = self._get_noised_known_composition(
                i, number_of_samples, device
            )

            # Denoise from t_{i+1} to t_i
            denoised_composition_i = self.generator.predictor_step(
                composition_ip1, i + 1, unit_cell, forces
            )

            composition_i = self._combine_noised_and_denoised_compositions(
                noised_composition_i, denoised_composition_i
            )

            for _ in range(self.generator.number_of_corrector_steps):
                composition_i = self.generator.corrector_step(
                    composition_i, i, unit_cell, forces
                )

            composition_ip1 = composition_i

        # apply the constraint one last time
        composition_i = self._apply_constraint(composition_i, device)

        return composition_i

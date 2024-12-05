import torch
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.sampling_constraint import \
    SamplingConstraint
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


class ConstrainedPredictorCorrectorAXLGenerator:
    """Constrained Predictor Corrector AXL Generator.

    This class constrains the input PC generator following a basic version of the inpainting algorithm
    presented in the paper
        "RePaint: Inpainting using Denoising Diffusion Probabilistic Models".
    """

    def __init__(
        self,
        generator: LangevinGenerator,
        sampling_constraint: SamplingConstraint,
    ):
        """Init method."""
        self.generator = generator
        self.sampling_constraint = sampling_constraint

        self.starting_free_diffusion_time_step = (
            self.sampling_constraint.sampling_constraint_parameters.starting_free_diffusion_time_step
        )

        self.constrained_atom_indices = (
            self.sampling_constraint.sampling_constraint_parameters.constrained_atom_indices
        )

        # Tensor dimensions: [number_of_time_steps, number_of_atoms, ...]
        self.constraint_compositions = self.sampling_constraint.constraint_compositions

        assert (
            self.generator.noise_parameters
            == self.sampling_constraint.sampling_constraint_parameters.noise_parameters
        ), "Noise parameters are inconsistent between generator and constraint."

        if hasattr(self.generator, "sample_trajectory_recorder"):
            self.sample_trajectory_recorder = self.generator.sample_trajectory_recorder

        self.number_of_atoms = self.generator.number_of_atoms
        self.all_atom_indices = torch.arange(self.number_of_atoms)
        self.num_classes = self.generator.num_classes

    def _apply_constraint(
        self, predicted_composition: AXL, time_index: int, device: torch.device
    ) -> AXL:
        """This method applies the constraints from the sampling_constraint."""
        if time_index > self.starting_free_diffusion_time_step:
            atom_indices = torch.arange(self.number_of_atoms)
        else:
            atom_indices = self.constrained_atom_indices

        constrained_x = predicted_composition.X.clone()
        constrained_a = predicted_composition.A.clone()

        constrained_x[:, atom_indices] = self.constraint_compositions.X[
            time_index, atom_indices
        ].to(device)
        constrained_a[:, atom_indices] = self.constraint_compositions.A[
            time_index, atom_indices
        ].to(device)

        constrained_composition = AXL(
            A=constrained_a,
            X=constrained_x,
            L=predicted_composition.L,
        )
        return constrained_composition

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
            # Denoise from t_{i+1} to t_i
            predicted_composition_i = self.generator.predictor_step(
                composition_ip1, i + 1, unit_cell, forces
            )
            composition_i = self._apply_constraint(predicted_composition_i, i, device)

            for _ in range(self.generator.number_of_corrector_steps):
                composition_i = self.generator.corrector_step(
                    composition_i, i, unit_cell, forces
                )

            composition_ip1 = composition_i

        # apply the constraint one last time
        composition_i = self._apply_constraint(composition_i, 0, device)

        return composition_i

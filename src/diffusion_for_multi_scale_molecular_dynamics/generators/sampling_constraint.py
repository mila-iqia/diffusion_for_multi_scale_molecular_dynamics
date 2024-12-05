import dataclasses
from dataclasses import dataclass
from pathlib import Path

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.loss import \
    D3PMLossCalculator
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from diffusion_for_multi_scale_molecular_dynamics.noisers.atom_types_noiser import \
    AtomTypesNoiser
from diffusion_for_multi_scale_molecular_dynamics.noisers.relative_coordinates_noiser import \
    RelativeCoordinatesNoiser
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    class_index_to_onehot


@dataclass
class SamplingConstraintParameters:
    """Dataclass describing how constraints should be created."""

    noise_parameters: NoiseParameters

    # number of atom types excluding MASK
    num_atom_types: int

    # the time step index at which free diffusion starts.
    # Should be equal or smaller to total_time_steps
    starting_free_diffusion_time_step: int

    # Indices of the atoms that should not diffuse freely.
    constrained_atom_indices: torch.Tensor

    # reference composition at t=0
    reference_composition: AXL

    # Small parameter to avoid dividing by zero in atom type probabilities.
    small_epsilon: float = 1.0e-12

    def __post_init__(self):
        """Post init method, to validate input."""
        assert (
            self.starting_free_diffusion_time_step
            <= self.noise_parameters.total_time_steps
        ), "Starting free diffusion time step should be smaller or equal to the total number of time steps."

        assert (
            self.num_atom_types >= 0
        ), "The number of atom types should be non-negative."

        assert (
            type(self.reference_composition.X) is torch.Tensor
        ), "the reference composition should be made of torch Tensors."
        assert (
            type(self.reference_composition.A) is torch.Tensor
        ), "the reference composition should be made of torch Tensors."
        assert (
            type(self.reference_composition.L) is torch.Tensor
        ), "the reference composition should be made of torch Tensors."

        assert len(self.reference_composition.X.shape) == 2, "X has the wrong shape."
        assert len(self.reference_composition.A.shape) == 1, "A has the wrong shape."

        (number_of_atoms,) = self.reference_composition.A.shape
        number_of_atoms_, _ = self.reference_composition.X.shape

        assert (
            number_of_atoms_ == number_of_atoms
        ), "The number of atoms in the reference composition should be consistent between X and A."

        indices_are_good = torch.logical_and(
            self.constrained_atom_indices >= 0,
            self.constrained_atom_indices < number_of_atoms,
        ).all()

        assert (
            indices_are_good
        ), "the constrained_atom_indices are inconsistent with the reference composition."


@dataclass
class SamplingConstraint:
    """Dataclass to hold the constraints for constrained sampling."""

    sampling_constraint_parameters: SamplingConstraintParameters

    # There should be one AXL reference per time step.
    # Tensors should have shape [time, natoms, ...]
    constraint_compositions: AXL

    def __post_init__(self):
        """Post init method, to validate input."""
        assert (
            type(self.constraint_compositions.X) is torch.Tensor
        ), "the reference compositions should torch Tensors."
        assert (
            type(self.constraint_compositions.A) is torch.Tensor
        ), "the reference compositions should torch Tensors."
        assert (
            type(self.constraint_compositions.L) is torch.Tensor
        ), "the reference compositions should torch Tensors."

        assert len(self.constraint_compositions.X.shape) == 3, "X has the wrong shape."
        assert len(self.constraint_compositions.A.shape) == 2, "A has the wrong shape."

        number_of_time_steps_, number_of_atoms = self.constraint_compositions.A.shape
        assert (
            number_of_time_steps_
            == self.sampling_constraint_parameters.noise_parameters.total_time_steps
        ), "The number of time steps in the constraint compositions should be total_number_of_time_steps."

        number_of_time_steps_, number_of_atoms_, _ = (
            self.constraint_compositions.X.shape
        )
        assert (
            number_of_time_steps_
            == self.sampling_constraint_parameters.noise_parameters.total_time_steps
        ), "The number of time steps in the constraint compositions should be total_number_of_time_steps."

        assert (
            number_of_atoms_ == number_of_atoms
        ), "The number of atoms in the constraint compositions should be consistent between X and A."


def create_sampling_constraint(
    sampling_constraint_parameters: SamplingConstraintParameters,
) -> SamplingConstraint:
    """Create sampling constraint."""
    total_time_steps = sampling_constraint_parameters.noise_parameters.total_time_steps
    # Account for the MASK class
    num_classes = sampling_constraint_parameters.num_atom_types + 1

    masked_atom_index = num_classes - 1

    sampler = NoiseScheduler(
        sampling_constraint_parameters.noise_parameters, num_classes=num_classes
    )
    noise, _ = sampler.get_all_sampling_parameters()

    relative_coordinates_noiser = RelativeCoordinatesNoiser()
    atom_type_noiser = AtomTypesNoiser()

    a_0 = sampling_constraint_parameters.reference_composition.A
    x_0 = sampling_constraint_parameters.reference_composition.X
    l_0 = sampling_constraint_parameters.reference_composition.L

    number_of_atoms, spatial_dimension = x_0.shape

    trajectory_a = torch.empty(total_time_steps, number_of_atoms, dtype=torch.int64)
    trajectory_x = torch.empty(total_time_steps, number_of_atoms, spatial_dimension)
    trajectory_l = torch.empty(total_time_steps, spatial_dimension, spatial_dimension)

    one_hot_a_0 = class_index_to_onehot(a_0, num_classes=num_classes)

    a_ip1 = masked_atom_index * torch.ones(number_of_atoms, dtype=torch.int64)
    one_hot_a_ip1 = class_index_to_onehot(a_ip1, num_classes=num_classes)

    coordinates_broadcasting = torch.ones_like(x_0)

    # Use the same Z-scores for all noising. This insures that all noised structures are "close" to each other.
    z_scores = relative_coordinates_noiser.get_gaussian_noise(x_0.shape)

    atom_is_unmasked = torch.zeros(number_of_atoms, dtype=torch.bool)

    for time_idx in torch.arange(total_time_steps - 1, 0, -1):
        sigmas_i = noise.sigma[time_idx] * coordinates_broadcasting
        x_i = relative_coordinates_noiser.get_noisy_relative_coordinates_sample_given_z_scores(
            x_0, sigmas_i, z_scores
        )

        q_matrices = einops.repeat(
            noise.q_matrix[time_idx],
            "n1 n2 -> natoms n1 n2",
            natoms=number_of_atoms,
        )

        q_bar_matrices = einops.repeat(
            noise.q_bar_matrix[time_idx],
            "n1 n2 -> natoms n1 n2",
            natoms=number_of_atoms,
        )
        q_bar_tm1_matrices = einops.repeat(
            noise.q_bar_tm1_matrix[time_idx],
            "n1 n2 -> natoms n1 n2",
            natoms=number_of_atoms,
        )

        atom_type_probabilities = (
            D3PMLossCalculator.get_q_atm1_given_at_and_a0(one_hot_a_0,
                                                          one_hot_a_ip1,
                                                          q_matrices,
                                                          q_bar_matrices,
                                                          q_bar_tm1_matrices,
                                                          sampling_constraint_parameters.small_epsilon))

        uniform_noise = atom_type_noiser.get_uniform_noise(one_hot_a_0.shape)

        a_i = (
            AtomTypesNoiser.get_noisy_atom_type_sample_from_uniform_variable_and_probabilities(atom_type_probabilities,
                                                                                               uniform_noise))

        a_i[atom_is_unmasked] = a_ip1[atom_is_unmasked]

        trajectory_a[time_idx] = a_i
        trajectory_x[time_idx] = x_i
        trajectory_l[time_idx] = l_0

        a_ip1 = a_i
        atom_is_unmasked = a_i != masked_atom_index

    trajectory_a[0] = a_0
    trajectory_x[0] = x_0
    trajectory_l[0] = l_0

    constraint_compositions = AXL(A=trajectory_a, X=trajectory_x, L=trajectory_l)

    sampling_constraint = SamplingConstraint(
        sampling_constraint_parameters=sampling_constraint_parameters,
        constraint_compositions=constraint_compositions,
    )

    return sampling_constraint


def write_sampling_constraint(
    sampling_constraint: SamplingConstraint, output_path: Path
):
    """Write sampling constraint.

    Args:
        sampling_constraint: object to write to file
        output_path: destination

    Returns:
        None.
    """
    # Write to disk as a dictionary to avoid conflicts if code changes.
    torch.save(dataclasses.asdict(sampling_constraint), output_path)


def read_sampling_constraint(output_path: Path) -> SamplingConstraint:
    """Read Sampling Constraint.

    Args:
        output_path: location of file on disk.

    Returns:
        sampling_constraint: object read from file.
    """
    sampling_constraint_dict = torch.load(output_path)

    sampling_constraint_parameters_dict = sampling_constraint_dict[
        "sampling_constraint_parameters"
    ]
    noise_parameters_dict = sampling_constraint_parameters_dict.pop("noise_parameters")
    noise_parameters = NoiseParameters(**noise_parameters_dict)

    sampling_constraint_parameters = SamplingConstraintParameters(
        noise_parameters=noise_parameters, **sampling_constraint_parameters_dict
    )

    constraint_compositions = sampling_constraint_dict["constraint_compositions"]

    return SamplingConstraint(
        sampling_constraint_parameters=sampling_constraint_parameters,
        constraint_compositions=constraint_compositions,
    )

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import einops
import torch

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

    sampler = NoiseScheduler(
        sampling_constraint_parameters.noise_parameters, num_classes=num_classes
    )
    noise, _ = sampler.get_all_sampling_parameters()

    relative_coordinates_noiser = RelativeCoordinatesNoiser()
    atom_type_noiser = AtomTypesNoiser()

    a_0 = sampling_constraint_parameters.reference_composition.A
    x_0 = sampling_constraint_parameters.reference_composition.X
    l_0 = sampling_constraint_parameters.reference_composition.L

    one_hot_a_0 = class_index_to_onehot(a_0, num_classes=num_classes)
    number_of_atoms = x_0.shape[0]

    coordinates_broadcasting = torch.ones_like(x_0)

    # Use the same Z-scores for all noising. This insures that all noised structures are "close" to each other.
    z_scores = RelativeCoordinatesNoiser._get_gaussian_noise(x_0.shape)

    list_x = [x_0]
    list_a = [a_0]
    list_l = [l_0]

    for time_idx in torch.arange(1, total_time_steps):
        sigmas_i = noise.sigma[time_idx] * coordinates_broadcasting
        q_bar_matrices_i = einops.repeat(
            noise.q_bar_matrix[time_idx],
            "n1 n2 -> natoms n1 n2",
            natoms=number_of_atoms,
        )

        x_i = relative_coordinates_noiser.get_noisy_relative_coordinates_sample_given_z_scores(
            x_0, sigmas_i, z_scores
        )

        a_i = atom_type_noiser.get_noisy_atom_types_sample(
            one_hot_a_0, q_bar_matrices_i
        )

        list_x.append(x_i)
        list_a.append(a_i)
        list_l.append(l_0)  # TODO : take care of L correctly.

    constraint_compositions = AXL(
        A=torch.stack(list_a, dim=0),
        X=torch.stack(list_x, dim=0),
        L=torch.stack(list_l, dim=0),
    )

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

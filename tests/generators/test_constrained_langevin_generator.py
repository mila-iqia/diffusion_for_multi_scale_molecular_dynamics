import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.constrained_langevin_generator import \
    ConstrainedPredictorCorrectorAXLGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.sampling_constraint import (
    SamplingConstraintParameters, create_sampling_constraint)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from tests.generators.test_langevin_generator import TestLangevinGenerator


class TestConstrainedLangevinGenerator(TestLangevinGenerator):

    @pytest.fixture(params=["full", "half"])
    def starting_free_diffusion_time_step(self, total_time_steps, request):
        if request.param == "full":
            return total_time_steps
        elif request.param == "half":
            return total_time_steps // 2

    @pytest.fixture()
    def constrained_atom_indices(self, number_of_atoms):
        number_of_constraints = number_of_atoms // 2
        return torch.randperm(number_of_atoms)[:number_of_constraints]

    @pytest.fixture()
    def reference_composition(
        self,
        number_of_atoms,
        spatial_dimension,
        num_atomic_classes,
        device,
    ):
        return AXL(
            A=torch.randint(0, num_atomic_classes, (number_of_atoms,)),
            X=map_relative_coordinates_to_unit_cell(
                torch.rand(number_of_atoms, spatial_dimension)
            ),
            L=torch.zeros(spatial_dimension, spatial_dimension),  # TODO placeholder
        )

    @pytest.fixture()
    def sampling_constraint_parameters(
        self,
        noise_parameters,
        num_atom_types,
        starting_free_diffusion_time_step,
        constrained_atom_indices,
        reference_composition,
    ):
        return SamplingConstraintParameters(
            noise_parameters=noise_parameters,
            num_atom_types=num_atom_types,
            starting_free_diffusion_time_step=starting_free_diffusion_time_step,
            constrained_atom_indices=constrained_atom_indices,
            reference_composition=reference_composition,
        )

    @pytest.fixture()
    def sampling_constraint(self, sampling_constraint_parameters):
        return create_sampling_constraint(sampling_constraint_parameters)

    @pytest.fixture()
    def random_compositions(
        self,
        number_of_samples,
        number_of_atoms,
        spatial_dimension,
        num_atomic_classes,
        device,
    ):
        return AXL(
            A=torch.randint(
                0,
                num_atomic_classes,
                (
                    number_of_samples,
                    number_of_atoms,
                ),
            ).to(device),
            X=map_relative_coordinates_to_unit_cell(
                torch.rand(number_of_samples, number_of_atoms, spatial_dimension)
            ).to(device),
            L=torch.zeros(spatial_dimension, spatial_dimension).to(
                device
            ),  # TODO placeholder
        )

    @pytest.fixture()
    def constrained_pc_generator(self, pc_generator, sampling_constraint):
        constrained_generator = ConstrainedPredictorCorrectorAXLGenerator(
            generator=pc_generator,
            sampling_constraint=sampling_constraint,
        )
        return constrained_generator

    @pytest.fixture()
    def constrained_samples(
        self, constrained_pc_generator, number_of_samples, device, unit_cell_sample
    ):
        samples = constrained_pc_generator.sample(
            number_of_samples, device, unit_cell_sample
        )
        return samples

    def test_constraints(
        self,
        constrained_samples,
        reference_composition,
        constrained_atom_indices,
        number_of_samples,
        device,
    ):
        reference_x = einops.repeat(
            reference_composition.X[constrained_atom_indices],
            "... -> n ...",
            n=number_of_samples,
        ).to(device)
        reference_a = einops.repeat(
            reference_composition.A[constrained_atom_indices],
            "... -> n ...",
            n=number_of_samples,
        ).to(device)

        torch.testing.assert_close(
            constrained_samples.X[:, constrained_atom_indices], reference_x
        )
        torch.testing.assert_close(
            constrained_samples.A[:, constrained_atom_indices], reference_a
        )

    def test_apply_constraint(
        self,
        constrained_pc_generator,
        total_time_steps,
        starting_free_diffusion_time_step,
        number_of_samples,
        number_of_atoms,
        random_compositions,
        constrained_atom_indices,
        sampling_constraint,
        device,
    ):

        reference_x = einops.repeat(
            sampling_constraint.constraint_compositions.X,
            "ntimes ... -> ntimes nsamples ...",
            nsamples=number_of_samples,
        ).to(device)
        reference_a = einops.repeat(
            sampling_constraint.constraint_compositions.A,
            "ntimes ... -> ntimes nsamples ...",
            nsamples=number_of_samples,
        ).to(device)

        for time_idx in torch.arange(total_time_steps - 1):
            if starting_free_diffusion_time_step < time_idx:
                atom_indices = torch.arange(number_of_atoms)
            else:
                atom_indices = constrained_atom_indices

            constrained_compositions = constrained_pc_generator._apply_constraint(
                random_compositions, time_idx, device
            )

            torch.testing.assert_close(
                constrained_compositions.X[:, atom_indices],
                reference_x[time_idx, :, atom_indices],
            )
            torch.testing.assert_close(
                constrained_compositions.A[:, atom_indices],
                reference_a[time_idx, :, atom_indices],
            )

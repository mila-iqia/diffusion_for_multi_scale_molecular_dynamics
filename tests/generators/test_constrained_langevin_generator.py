import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.constrained_langevin_generator import \
    ConstrainedPredictorCorrectorAXLGenerator
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from tests.generators.test_langevin_generator import TestLangevinGenerator


class TestConstrainedLangevinGenerator(TestLangevinGenerator):

    @pytest.fixture()
    def reference_composition(
        self,
        number_of_atoms,
        spatial_dimension,
        num_atomic_classes,
        device,
    ):
        return AXL(
            A=torch.randint(0, num_atomic_classes, (number_of_atoms,)).to(device),
            X=map_relative_coordinates_to_unit_cell(
                torch.rand(number_of_atoms, spatial_dimension)
            ).to(device),
            L=torch.zeros(spatial_dimension * (spatial_dimension - 1)).to(
                device
            ),  # TODO placeholder
        )

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
            L=torch.zeros(spatial_dimension * (spatial_dimension - 1)).to(
                device
            ),  # TODO placeholder
        )

    @pytest.fixture()
    def constrained_atom_indices(self, number_of_atoms, device):
        number_of_constraints = number_of_atoms // 2
        return torch.randperm(number_of_atoms)[:number_of_constraints].to(device)

    @pytest.fixture()
    def constrained_pc_generator(
        self, pc_generator, reference_composition, constrained_atom_indices
    ):
        constrained_generator = ConstrainedPredictorCorrectorAXLGenerator(
            generator=pc_generator,
            reference_composition=reference_composition,
            constrained_atom_indices=constrained_atom_indices,
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
    ):
        reference_x = einops.repeat(
            reference_composition.X[constrained_atom_indices],
            "... -> n ...",
            n=number_of_samples,
        )
        reference_a = einops.repeat(
            reference_composition.A[constrained_atom_indices],
            "... -> n ...",
            n=number_of_samples,
        )

        torch.testing.assert_close(
            constrained_samples.X[:, constrained_atom_indices], reference_x
        )
        torch.testing.assert_close(
            constrained_samples.A[:, constrained_atom_indices], reference_a
        )

    def test_apply_constraint(
        self,
        constrained_pc_generator,
        number_of_samples,
        random_compositions,
        reference_composition,
        constrained_atom_indices,
        device,
    ):

        constrained_compositions = constrained_pc_generator._apply_constraint(
            random_compositions, device
        )

        reference_x = einops.repeat(
            reference_composition.X[constrained_atom_indices],
            "... -> n ...",
            n=number_of_samples,
        )
        reference_a = einops.repeat(
            reference_composition.A[constrained_atom_indices],
            "... -> n ...",
            n=number_of_samples,
        )

        torch.testing.assert_close(
            constrained_compositions.X[:, constrained_atom_indices], reference_x
        )
        torch.testing.assert_close(
            constrained_compositions.A[:, constrained_atom_indices], reference_a
        )

    def test_combine_noised_and_denoised_compositions(
        self,
        constrained_pc_generator,
        constrained_atom_indices,
        number_of_samples,
        number_of_atoms,
        spatial_dimension,
        device,
    ) -> AXL:

        noised_mask = torch.zeros(number_of_atoms, dtype=torch.bool).to(device)
        noised_mask[constrained_atom_indices] = True

        noised_compositions = AXL(
            A=torch.zeros(number_of_samples, number_of_atoms).to(device),
            X=torch.zeros(number_of_samples, number_of_atoms, spatial_dimension).to(
                device
            ),
            L=0.0,
        )

        denoised_compositions = AXL(
            A=torch.ones(number_of_samples, number_of_atoms).to(device),
            X=torch.ones(number_of_samples, number_of_atoms, spatial_dimension).to(
                device
            ),
            L=0.0,
        )

        combined_compositions = (
            constrained_pc_generator._combine_noised_and_denoised_compositions(
                noised_compositions, denoised_compositions
            )
        )

        assert (combined_compositions.X[:, noised_mask] == 0.0).all()
        assert (combined_compositions.X[:, ~noised_mask] == 1.0).all()
        assert (combined_compositions.A[:, noised_mask] == 0.0).all()
        assert (combined_compositions.A[:, ~noised_mask] == 1.0).all()

import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.constrained_langevin_generator import \
    ConstrainedLangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.sampling_constraint import \
    SamplingConstraint
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from tests.fake_data_utils import generate_random_string
from tests.generators.test_langevin_generator import TestLangevinGenerator


class TestConstrainedLangevinGenerator(TestLangevinGenerator):

    @pytest.fixture()
    def elements(self, num_atom_types):
        return [generate_random_string(size=4) for _ in range(num_atom_types)]

    @pytest.fixture()
    def number_of_constraints(self, number_of_atoms):
        return number_of_atoms // 2

    @pytest.fixture()
    def constrained_indices(self, number_of_atoms, number_of_constraints):
        return torch.randperm(number_of_atoms)[:number_of_constraints]

    @pytest.fixture()
    def constrained_relative_coordinates(self, number_of_constraints, spatial_dimension):
        return torch.rand(number_of_constraints, spatial_dimension)

    @pytest.fixture()
    def constrained_atom_types(self, number_of_constraints, elements):
        return torch.randint(0, len(elements), (number_of_constraints,))

    @pytest.fixture()
    def sampling_constraint(self, elements, constrained_relative_coordinates,
                            constrained_atom_types, constrained_indices):
        return SamplingConstraint(elements=elements,
                                  constrained_relative_coordinates=constrained_relative_coordinates,
                                  constrained_atom_types=constrained_atom_types,
                                  constrained_indices=constrained_indices)

    @pytest.fixture()
    def pc_generator(self, noise_parameters, sampling_parameters, axl_network, sampling_constraint):
        generator = ConstrainedLangevinGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=axl_network,
            sampling_constraints=sampling_constraint
        )
        return generator

    @pytest.fixture()
    def composition(
        self,
        number_of_samples,
        number_of_atoms,
        spatial_dimension,
        num_atom_types,
        device,
    ):
        return AXL(
            A=torch.randint(
                0, num_atom_types + 1, (number_of_samples, number_of_atoms)
            ).to(device),
            X=torch.rand(number_of_samples, number_of_atoms, spatial_dimension).to(
                device
            ),
            L=torch.rand(
                number_of_samples, spatial_dimension * (spatial_dimension - 1)
            ).to(
                device
            ),
        )

    @pytest.fixture()
    def repaint_is_used(self):
        # Since we are repainting the atom types, there can be many changes in one pass, and
        # masking can be changed.
        return True

    def test_apply_constraint(
        self, pc_generator, composition, sampling_constraint, device
    ):
        batch_size = composition.X.shape[0]
        constrained_composition = pc_generator._apply_constraint(composition, device)

        constrained_a = einops.repeat(sampling_constraint.constrained_atom_types.to(device),
                                      "n -> b n",
                                      b=batch_size,
                                      )

        constrained_x = einops.repeat(sampling_constraint.constrained_relative_coordinates.to(device),
                                      "n d -> b n d",
                                      b=batch_size,
                                      )

        torch.testing.assert_close(constrained_a, constrained_composition.A[:, sampling_constraint.constrained_indices])
        torch.testing.assert_close(constrained_x, constrained_composition.X[:, sampling_constraint.constrained_indices])

    def test_get_composition_0_known(self, pc_generator, number_of_samples, sampling_constraint, device) -> AXL:
        composition0_known = pc_generator._get_composition_0_known(number_of_samples, device)

        batch_size = composition0_known.X.shape[0]

        constrained_a = einops.repeat(sampling_constraint.constrained_atom_types.to(device),
                                      "n -> b n",
                                      b=batch_size,
                                      )

        constrained_x = einops.repeat(sampling_constraint.constrained_relative_coordinates.to(device),
                                      "n d -> b n d",
                                      b=batch_size,
                                      )

        torch.testing.assert_close(constrained_a, composition0_known.A[:, sampling_constraint.constrained_indices])
        torch.testing.assert_close(constrained_x, composition0_known.X[:, sampling_constraint.constrained_indices])

    @pytest.mark.skipif(True, reason="This test is not appropriate because predictor_step has been modified.")
    def test_predictor_step_relative_coordinates_and_lattice(self):
        pass

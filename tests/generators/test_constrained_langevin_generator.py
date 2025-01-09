import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.constrained_langevin_generator import (
    ConstrainedLangevinGenerator, ConstrainedLangevinGeneratorParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from tests.generators.test_langevin_generator import TestLangevinGenerator


class TestConstrainedLangevinGenerator(TestLangevinGenerator):

    @pytest.fixture()
    def constrained_relative_coordinates(self, number_of_atoms, spatial_dimension):
        number_of_constraints = number_of_atoms // 2
        return torch.rand(number_of_constraints, spatial_dimension).numpy()

    @pytest.fixture()
    def sampling_parameters(
        self,
        number_of_atoms,
        spatial_dimension,
        number_of_samples,
        number_of_corrector_steps,
        unit_cell_size,
        constrained_relative_coordinates,
        num_atom_types,
    ):
        sampling_parameters = ConstrainedLangevinGeneratorParameters(
            number_of_corrector_steps=number_of_corrector_steps,
            number_of_atoms=number_of_atoms,
            number_of_samples=number_of_samples,
            spatial_dimension=spatial_dimension,
            constrained_relative_coordinates=constrained_relative_coordinates,
            num_atom_types=num_atom_types,
        )

        return sampling_parameters

    @pytest.fixture()
    def pc_generator(self, noise_parameters, sampling_parameters, axl_network):
        generator = ConstrainedLangevinGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=axl_network,
        )

        return generator

    @pytest.fixture()
    def axl(
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
            ),  # TODO placeholder
        )

    def test_apply_constraint(
        self, pc_generator, axl, constrained_relative_coordinates, device
    ):
        batch_size = axl.X.shape[0]
        original_x = torch.clone(axl.X)
        pc_generator._apply_constraint(axl, device)

        number_of_constraints = len(constrained_relative_coordinates)

        constrained_x = einops.repeat(
            torch.from_numpy(constrained_relative_coordinates).to(device),
            "n d -> b n d",
            b=batch_size,
        )

        torch.testing.assert_close(axl.X[:, :number_of_constraints], constrained_x)
        torch.testing.assert_close(
            axl.X[:, number_of_constraints:], original_x[:, number_of_constraints:]
        )

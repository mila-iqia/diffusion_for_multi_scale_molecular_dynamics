import einops
import pytest
import torch

from crystal_diffusion.generators.constrained_langevin_dynamics_generator import (
    ConstrainedLangevinGenerator, ConstrainedLangevinGeneratorParameters)
from tests.generators.test_langevin_dynamics import \
    TestAnnealedLangevinDynamics


class TestConstrainedAnnealedLangevinDyamics(TestAnnealedLangevinDynamics):

    @pytest.fixture()
    def constrained_relative_coordinates(self, number_of_atoms, spatial_dimension):
        number_of_constraints = number_of_atoms // 2
        return torch.rand(number_of_constraints, spatial_dimension).numpy()

    @pytest.fixture()
    def sampling_parameters(self, number_of_atoms, spatial_dimension, number_of_samples,
                            number_of_corrector_steps, unit_cell_size, constrained_relative_coordinates):
        sampling_parameters = ConstrainedLangevinGeneratorParameters(
            number_of_corrector_steps=number_of_corrector_steps,
            number_of_atoms=number_of_atoms,
            number_of_samples=number_of_samples,
            cell_dimensions=spatial_dimension * [unit_cell_size],
            spatial_dimension=spatial_dimension,
            constrained_relative_coordinates=constrained_relative_coordinates)

        return sampling_parameters

    @pytest.fixture()
    def pc_generator(self, noise_parameters, sampling_parameters, sigma_normalized_score_network):
        generator = ConstrainedLangevinGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            sigma_normalized_score_network=sigma_normalized_score_network)

        return generator

    @pytest.fixture()
    def x(self, number_of_samples, number_of_atoms, spatial_dimension, device):
        return torch.rand(number_of_samples, number_of_atoms, spatial_dimension).to(device)

    def test_apply_constraint(self, pc_generator, x, constrained_relative_coordinates, device):
        batch_size = x.shape[0]
        original_x = torch.clone(x)
        pc_generator._apply_constraint(x, device)

        number_of_constraints = len(constrained_relative_coordinates)

        constrained_x = einops.repeat(torch.from_numpy(constrained_relative_coordinates).to(device),
                                      "n d -> b n d", b=batch_size)

        torch.testing.assert_close(x[:, :number_of_constraints], constrained_x)
        torch.testing.assert_close(x[:, number_of_constraints:], original_x[:, number_of_constraints:])

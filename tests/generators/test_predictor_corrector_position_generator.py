import pytest
import torch

from crystal_diffusion.generators.predictor_corrector_position_generator import \
    PredictorCorrectorPositionGenerator
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from tests.generators.conftest import BaseTestGenerator


class FakePCGenerator(PredictorCorrectorPositionGenerator):
    """A dummy PC generator for the purpose of testing."""

    def __init__(
        self,
        number_of_discretization_steps: int,
        number_of_corrector_steps: int,
        spatial_dimension: int,
        initial_sample: torch.Tensor,
    ):
        super().__init__(number_of_discretization_steps, number_of_corrector_steps, spatial_dimension)
        self.initial_sample = initial_sample

    def initialize(self, number_of_samples: int):
        return self.initial_sample

    def predictor_step(self, x_ip1: torch.Tensor, ip1: int, unit_cell: torch.Tensor, forces: torch.Tensor
                       ) -> torch.Tensor:
        return 1.2 * x_ip1 + 3.4 + ip1 / 111.0

    def corrector_step(self, x_i: torch.Tensor, i: int, unit_cell: torch.Tensor, forces: torch.Tensor
                       ) -> torch.Tensor:
        return 0.56 * x_i + 7.89 + i / 117.0


@pytest.mark.parametrize("number_of_discretization_steps", [1, 5, 10])
@pytest.mark.parametrize("number_of_corrector_steps", [0, 1, 2])
class TestPredictorCorrectorPositionGenerator(BaseTestGenerator):
    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(1234567)

    @pytest.fixture
    def initial_sample(self, number_of_samples, number_of_atoms, spatial_dimension):
        return torch.rand(number_of_samples, number_of_atoms, spatial_dimension)

    @pytest.fixture
    def generator(
        self, number_of_discretization_steps, number_of_corrector_steps, spatial_dimension, initial_sample
    ):
        generator = FakePCGenerator(
            number_of_discretization_steps, number_of_corrector_steps, spatial_dimension, initial_sample
        )
        return generator

    @pytest.fixture
    def expected_samples(
        self,
        generator,
        initial_sample,
        number_of_discretization_steps,
        number_of_corrector_steps,
        unit_cell_sample,
    ):
        list_i = list(range(number_of_discretization_steps))
        list_i.reverse()
        list_j = list(range(number_of_corrector_steps))

        noisy_sample = map_relative_coordinates_to_unit_cell(initial_sample)
        x_ip1 = noisy_sample
        for i in list_i:
            xi = map_relative_coordinates_to_unit_cell(generator.predictor_step(x_ip1, i + 1, unit_cell_sample,
                                                                                torch.zeros_like(x_ip1)))
            for _ in list_j:
                xi = map_relative_coordinates_to_unit_cell(generator.corrector_step(xi, i, unit_cell_sample,
                                                                                    torch.zeros_like(xi)))
            x_ip1 = xi
        return xi

    def test_sample(self, generator, number_of_samples, expected_samples, unit_cell_sample):
        computed_samples = generator.sample(number_of_samples, torch.device('cpu'), unit_cell_sample)
        torch.testing.assert_close(expected_samples, computed_samples)

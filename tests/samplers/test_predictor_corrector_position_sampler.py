import pytest
import torch

from crystal_diffusion.samplers.noisy_position_sampler import \
    map_positions_to_unit_cell
from crystal_diffusion.samplers.predictor_corrector_position_sampler import \
    PredictorCorrectorPositionSampler


class FakePCSampler(PredictorCorrectorPositionSampler):
    """A dummy PC sampler for the purpose of testing."""

    def __init__(
        self,
        number_of_discretization_steps: int,
        number_of_corrector_steps: int,
        initial_sample: torch.Tensor,
    ):
        super().__init__(number_of_discretization_steps, number_of_corrector_steps)
        self.initial_sample = initial_sample

    def initialize(self, number_of_samples: int):
        return self.initial_sample

    def predictor_step(self, x_ip1: torch.Tensor, ip1: int) -> torch.Tensor:
        return 1.2 * x_ip1 + 3.4 + ip1 / 111.

    def corrector_step(self, x_i: torch.Tensor, i: int) -> torch.Tensor:
        return 0.56 * x_i + 7.89 + i / 117.


@pytest.mark.parametrize("number_of_samples", [4])
@pytest.mark.parametrize("number_of_atoms", [8])
@pytest.mark.parametrize("spatial_dimension", [2, 3])
@pytest.mark.parametrize("number_of_discretization_steps", [1, 5, 10])
@pytest.mark.parametrize("number_of_corrector_steps", [0, 1, 2])
class TestPredictorCorrectorPositionSampler:
    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(1234567)

    @pytest.fixture
    def initial_sample(self, number_of_samples, number_of_atoms, spatial_dimension):
        return torch.rand(number_of_samples, number_of_atoms, spatial_dimension)

    @pytest.fixture
    def sampler(
        self, number_of_discretization_steps, number_of_corrector_steps, initial_sample
    ):
        sampler = FakePCSampler(
            number_of_discretization_steps, number_of_corrector_steps, initial_sample
        )
        return sampler

    @pytest.fixture
    def expected_samples(
        self,
        sampler,
        initial_sample,
        number_of_discretization_steps,
        number_of_corrector_steps,
    ):
        list_i = list(range(number_of_discretization_steps))
        list_i.reverse()
        list_j = list(range(number_of_corrector_steps))

        noisy_sample = map_positions_to_unit_cell(initial_sample)
        x_ip1 = noisy_sample
        for i in list_i:
            xi = map_positions_to_unit_cell(sampler.predictor_step(x_ip1, i + 1))
            for _ in list_j:
                xi = map_positions_to_unit_cell(sampler.corrector_step(xi, i))
            x_ip1 = xi
        return xi

    def test_sample(self, sampler, number_of_samples, expected_samples):
        computed_samples = sampler.sample(number_of_samples)
        torch.testing.assert_allclose(expected_samples, computed_samples)

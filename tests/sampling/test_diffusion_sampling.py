import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    CARTESIAN_POSITIONS, RELATIVE_COORDINATES, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_positions_from_coordinates
from src.diffusion_for_multi_scale_molecular_dynamics.generators.position_generator import (
    PositionGenerator, SamplingParameters)
from src.diffusion_for_multi_scale_molecular_dynamics.sampling.diffusion_sampling import \
    create_batch_of_samples


class DummyGenerator(PositionGenerator):
    def __init__(self, relative_coordinates):
        self._relative_coordinates = relative_coordinates
        self._counter = 0

    def initialize(self, number_of_samples: int):
        pass

    def sample(
        self, number_of_samples: int, device: torch.device, unit_cell: torch.Tensor
    ) -> torch.Tensor:
        self._counter += number_of_samples
        return self._relative_coordinates[
            self._counter - number_of_samples: self._counter
        ]


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def number_of_samples():
    return 16


@pytest.fixture
def number_of_atoms():
    return 8


@pytest.fixture
def spatial_dimensions():
    return 3


@pytest.fixture
def relative_coordinates(number_of_samples, number_of_atoms, spatial_dimensions):
    return torch.rand(number_of_samples, number_of_atoms, spatial_dimensions)


@pytest.fixture
def cell_dimensions(spatial_dimensions):
    return list((10 * torch.rand(spatial_dimensions)).numpy())


@pytest.fixture
def generator(relative_coordinates):
    return DummyGenerator(relative_coordinates)


@pytest.fixture
def sampling_parameters(
    spatial_dimensions, number_of_atoms, number_of_samples, cell_dimensions
):
    return SamplingParameters(
        algorithm="dummy",
        spatial_dimension=spatial_dimensions,
        number_of_atoms=number_of_atoms,
        number_of_samples=number_of_samples,
        sample_batchsize=2,
        cell_dimensions=cell_dimensions,
    )


def test_create_batch_of_samples(
    generator, sampling_parameters, device, relative_coordinates, cell_dimensions
):
    computed_samples = create_batch_of_samples(generator, sampling_parameters, device)

    batch_size = computed_samples[UNIT_CELL].shape[0]

    expected_basis_vectors = einops.repeat(
        torch.diag(torch.tensor(cell_dimensions)), "d1 d2 -> b d1 d2", b=batch_size
    )

    expected_cartesian_coordinates = get_positions_from_coordinates(
        relative_coordinates, expected_basis_vectors
    )

    torch.testing.assert_allclose(
        computed_samples[RELATIVE_COORDINATES], relative_coordinates
    )
    torch.testing.assert_allclose(computed_samples[UNIT_CELL], expected_basis_vectors)
    torch.testing.assert_allclose(
        computed_samples[CARTESIAN_POSITIONS], expected_cartesian_coordinates
    )

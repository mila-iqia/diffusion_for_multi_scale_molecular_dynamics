import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import (
    AXLGenerator, SamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, AXL_COMPOSITION, CARTESIAN_POSITIONS)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates,
    map_lattice_parameters_to_unit_cell_vectors)
from src.diffusion_for_multi_scale_molecular_dynamics.sampling.diffusion_sampling import \
    create_batch_of_samples


class DummyGenerator(AXLGenerator):
    def __init__(self, relative_coordinates, lattice_parameters):
        self._relative_coordinates = relative_coordinates
        self._lattice_parameters = lattice_parameters
        self._counter = 0

    def initialize(self, number_of_samples: int):
        pass

    def sample(
        self,
        number_of_samples: int,
        device: torch.device,
    ) -> AXL:
        self._counter += number_of_samples
        rel_coordinates = self._relative_coordinates[
            self._counter - number_of_samples:self._counter
        ]
        lattice_parameters = self._lattice_parameters[
            self._counter - number_of_samples:self._counter
        ]
        return AXL(
            A=torch.zeros_like(rel_coordinates[..., 0]).long(),
            X=rel_coordinates,
            L=lattice_parameters,
        )


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
def num_atom_types():
    return 4


@pytest.fixture
def relative_coordinates(number_of_samples, number_of_atoms, spatial_dimensions):
    return torch.rand(number_of_samples, number_of_atoms, spatial_dimensions)


@pytest.fixture
def lattice_parameters(number_of_samples, spatial_dimensions):
    num_lattice_parameters = int(spatial_dimensions * (spatial_dimensions + 1) / 2)
    return torch.randn(number_of_samples, num_lattice_parameters)


@pytest.fixture
def generator(relative_coordinates, lattice_parameters):
    return DummyGenerator(relative_coordinates, lattice_parameters)


@pytest.fixture
def sampling_parameters(
    spatial_dimensions,
    number_of_atoms,
    number_of_samples,
    num_atom_types,
):
    return SamplingParameters(
        algorithm="dummy",
        spatial_dimension=spatial_dimensions,
        number_of_atoms=number_of_atoms,
        number_of_samples=number_of_samples,
        sample_batchsize=2,
        num_atom_types=num_atom_types,
    )


def test_create_batch_of_samples(
    generator, sampling_parameters, device, relative_coordinates, lattice_parameters
):
    computed_samples = create_batch_of_samples(generator, sampling_parameters, device)

    expected_basis_vectors = map_lattice_parameters_to_unit_cell_vectors(
        lattice_parameters
    )

    expected_cartesian_coordinates = get_positions_from_coordinates(
        relative_coordinates, expected_basis_vectors
    )

    torch.testing.assert_allclose(
        computed_samples[AXL_COMPOSITION].X, relative_coordinates
    )
    torch.testing.assert_allclose(
        computed_samples[AXL_COMPOSITION].L, lattice_parameters
    )
    torch.testing.assert_allclose(
        computed_samples[CARTESIAN_POSITIONS], expected_cartesian_coordinates
    )

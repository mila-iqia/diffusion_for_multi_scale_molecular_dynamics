import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import \
    SigmaCalculator
from tests.regularizers.differentiable_score_network import (
    DifferentiableScoreNetwork, DifferentiableScoreNetworkParameters)


class BaseTestRegularizer:

    @pytest.fixture(scope="class", autouse=True)
    def set_seed(self):
        """Set the random seed."""
        torch.manual_seed(34534234)

    @pytest.fixture()
    def sigma_min(self):
        return 0.001

    @pytest.fixture()
    def sigma_max(self):
        return 0.2

    @pytest.fixture()
    def number_of_atoms(self):
        return 4

    @pytest.fixture()
    def num_atom_types(self):
        return 1

    @pytest.fixture()
    def spatial_dimension(self):
        return 3

    @pytest.fixture()
    def batch_size(self):
        return 16

    @pytest.fixture()
    def relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension):
        return torch.rand(batch_size, number_of_atoms, spatial_dimension)

    @pytest.fixture()
    def times(self, batch_size):
        return torch.rand(batch_size, 1)

    @pytest.fixture()
    def sigmas(self, sigma_min, sigma_max, times):
        return SigmaCalculator(sigma_min=sigma_min, sigma_max=sigma_max).get_sigma(
            times
        )

    @pytest.fixture()
    def atom_types(self, batch_size, number_of_atoms):
        return torch.zeros(batch_size, number_of_atoms, dtype=torch.int64)

    @pytest.fixture()
    def cell_dimensions(self, spatial_dimension):
        acell = 5.0
        return acell * torch.ones(spatial_dimension)

    @pytest.fixture()
    def lattice_parameters(self, batch_size, spatial_dimension, cell_dimensions):
        lattice_params = torch.zeros(batch_size, int(spatial_dimension * (spatial_dimension + 1) / 2))
        lattice_params[:, :spatial_dimension] = cell_dimensions
        return lattice_params

    @pytest.fixture()
    def augmented_batch(
        self, relative_coordinates, times, sigmas, atom_types, lattice_parameters
    ):
        forces = torch.zeros_like(relative_coordinates)
        composition = AXL(A=atom_types, X=relative_coordinates, L=lattice_parameters)

        batch = {
            NOISY_AXL_COMPOSITION: composition,
            NOISE: sigmas,
            TIME: times,
            CARTESIAN_FORCES: forces,
        }
        return batch

    @pytest.fixture()
    def score_parameters(
        self, number_of_atoms, num_atom_types, spatial_dimension, sigma_min, sigma_max
    ):
        score_parameters = DifferentiableScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            num_atom_types=num_atom_types,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        return score_parameters

    @pytest.fixture()
    def score_network(self, score_parameters):
        return DifferentiableScoreNetwork(score_parameters)

    def test_compute_weighted_regularizer_loss(
        self, regularizer, score_network, augmented_batch
    ):

        # Smoke test that the method runs.
        _ = regularizer.compute_weighted_regularizer_loss(
            score_network=score_network,
            augmented_batch=augmented_batch,
            current_epoch=0,
        )

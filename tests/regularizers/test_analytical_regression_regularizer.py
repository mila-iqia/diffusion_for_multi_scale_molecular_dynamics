import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import \
    AnalyticalScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.regularizers.analytical_regression_regularizer import (
    AnalyticalRegressionRegularizer, AnalyticalRegressionRegularizerParameters)
from tests.regularizers.conftest import BaseTestRegularizer


class TestAnalyticalRegressionRegularizer(BaseTestRegularizer):

    @pytest.fixture()
    def maximum_number_of_steps(self):
        return 5

    @pytest.fixture()
    def total_time_steps(
        self,
    ):
        return 10

    @pytest.fixture()
    def noise_parameters(self, sigma_min, sigma_max, total_time_steps):
        return NoiseParameters(
            total_time_steps=total_time_steps, sigma_min=sigma_min, sigma_max=sigma_max
        )

    @pytest.fixture()
    def sampling_parameters(
        self, num_atom_types, number_of_atoms, batch_size, cell_dimensions
    ):
        return PredictorCorrectorSamplingParameters(
            number_of_corrector_steps=0,
            num_atom_types=num_atom_types,
            cell_dimensions=list(cell_dimensions.numpy()),
            number_of_atoms=number_of_atoms,
            number_of_samples=batch_size,
        )

    @pytest.fixture()
    def regularizer_parameters(
        self, number_of_atoms, spatial_dimension, num_atom_types
    ):

        coords = torch.rand(number_of_atoms, spatial_dimension)
        equilibrium_relative_coordinates = list(list(x) for x in coords.numpy())

        params = AnalyticalScoreNetworkParameters(number_of_atoms=number_of_atoms,
                                                  equilibrium_relative_coordinates=equilibrium_relative_coordinates,
                                                  kmax=5,
                                                  sigma_d=0.01,
                                                  num_atom_types=num_atom_types,
                                                  use_permutation_invariance=False)

        return AnalyticalRegressionRegularizerParameters(analytical_score_network_parameters=params)

    @pytest.fixture()
    def regularizer(self, regularizer_parameters, device):
        return AnalyticalRegressionRegularizer(regularizer_parameters, device=device)

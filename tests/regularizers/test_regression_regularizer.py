import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import \
    AnalyticalScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.equivariant_analytical_score_network import \
    EquivariantAnalyticalScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.regularizers.regression_regularizer import (
    RegressionRegularizer, RegressionRegularizerParameters)
from tests.regularizers.conftest import BaseTestRegularizer


class TestRegressionRegularizer(BaseTestRegularizer):

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

    @pytest.fixture(params=[True, False])
    def use_equivariant_analytical_score_network(self, request):
        return request.param

    @pytest.fixture()
    def regularizer_parameters(
        self, number_of_atoms, spatial_dimension, num_atom_types, use_equivariant_analytical_score_network
    ):
        coords = torch.rand(number_of_atoms, spatial_dimension)
        equilibrium_relative_coordinates = list(list(x) for x in coords.numpy())

        common_params_dict = dict(number_of_atoms=number_of_atoms,
                                  equilibrium_relative_coordinates=equilibrium_relative_coordinates,
                                  kmax=5,
                                  sigma_d=0.01,
                                  num_atom_types=num_atom_types)
        if use_equivariant_analytical_score_network:
            params = EquivariantAnalyticalScoreNetworkParameters(**common_params_dict)
        else:
            params = AnalyticalScoreNetworkParameters(**common_params_dict, use_permutation_invariance=False)

        return RegressionRegularizerParameters(score_network_parameters=params)

    @pytest.fixture()
    def regularizer(self, regularizer_parameters, device):
        return RegressionRegularizer(regularizer_parameters).to(device)

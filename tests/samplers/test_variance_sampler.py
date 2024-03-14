import pytest
import torch

from crystal_diffusion.samplers.time_sampler import TimeParameters, TimeSampler
from crystal_diffusion.samplers.variance_sampler import (
    ExplodingVarianceSampler, VarianceParameters)


@pytest.mark.parametrize('total_time_steps', [3, 10, 17])
class TestExplodingVarianceSampler:
    @pytest.fixture()
    def time_parameters(self, total_time_steps):
        time_parameters = TimeParameters(total_time_steps=total_time_steps, random_seed=0)
        return time_parameters

    @pytest.fixture()
    def time_sampler(self, time_parameters):
        return TimeSampler(time_parameters)

    @pytest.fixture()
    def variance_parameters(self, time_sampler):
        return VarianceParameters()

    @pytest.fixture()
    def variance_sampler(self, variance_parameters, time_sampler):
        return ExplodingVarianceSampler(variance_parameters=variance_parameters,
                                        time_sampler=time_sampler)

    @pytest.fixture()
    def expected_variances(self, variance_parameters, time_sampler, time_parameters):

        times = torch.linspace(0., 1., time_parameters.total_time_steps)

        smin = variance_parameters.sigma_min
        smax = variance_parameters.sigma_max

        sigmas = smin**(1. - times) * smax ** times
        variances = sigmas**2
        return variances

    @pytest.fixture()
    def indices(self, time_sampler, shape):
        return time_sampler.get_random_time_step_indices(shape)

    @pytest.fixture()
    def expected_variances_by_index(self, expected_variances, indices):
        result = []
        for i in indices.flatten():
            result.append(expected_variances[i])
        return torch.tensor(result).reshape(indices.shape)

    def test_sigma_square_array(self, variance_sampler, expected_variances):
        assert torch.all(torch.isclose(variance_sampler._sigma_square_array, expected_variances))

    def test_g_square_array(self, variance_sampler, expected_variances):

        expected_g_squared_array = [float('nan')]
        for sigma2_t, sigma2_tm1 in zip(expected_variances[1:], expected_variances[:-1]):
            g2 = sigma2_t - sigma2_tm1
            expected_g_squared_array.append(g2)

        expected_g_squared_array = torch.tensor(expected_g_squared_array)

        assert torch.isnan(variance_sampler._g_square_array[0])
        assert torch.all(torch.isclose(variance_sampler._g_square_array[1:], expected_g_squared_array[1:]))

    @pytest.mark.parametrize('shape', [(100,), (3, 8), (1, 2, 3)])
    def test_get_variances(self, variance_sampler, indices, expected_variances_by_index):
        computed_variances = variance_sampler.get_variances(indices)
        assert torch.all(torch.isclose(computed_variances, expected_variances_by_index))

    def test_get_g_squared_factors(self, variance_sampler, expected_variances):

        indices = []
        expected_g_squared_factors = []
        previous_variance = expected_variances[0]

        for index, current_variance in enumerate(expected_variances[1:], 1):
            indices.append(index)
            current_variance = expected_variances[index]
            g2 = current_variance - previous_variance
            expected_g_squared_factors.append(g2)
            previous_variance = current_variance
        expected_g_squared_factors = torch.tensor(expected_g_squared_factors)

        indices = torch.tensor(indices)
        computed_g_squared_factors = variance_sampler.get_g_squared_factors(indices)

        assert torch.all(torch.isclose(computed_g_squared_factors, expected_g_squared_factors))

import pytest
import torch

from crystal_diffusion.samplers.time_sampler import TimeParameters, TimeSampler


@pytest.mark.parametrize('total_time_steps', [3, 10, 17])
class TestTimeSampler:
    @pytest.fixture()
    def time_parameters(self, total_time_steps):
        time_parameters = TimeParameters(total_time_steps=total_time_steps, random_seed=0)
        return time_parameters

    @pytest.fixture()
    def time_sampler(self, time_parameters):
        return TimeSampler(time_parameters)

    @pytest.mark.parametrize('shape', [(100,), (3, 8), (1, 2, 3)])
    def test_random_time_step_indices(self, time_sampler, shape, total_time_steps):
        computed_time_step_indices = time_sampler.get_random_time_step_indices(shape)
        assert computed_time_step_indices.shape == shape
        assert torch.all(computed_time_step_indices > 0)
        assert torch.all(computed_time_step_indices <= total_time_steps)

    @pytest.mark.parametrize('shape', [(100,), (3, 8), (1, 2, 3)])
    def test_random_time_steps(self, time_sampler, shape, total_time_steps):
        indices = time_sampler.get_random_time_step_indices(shape)
        computed_time_steps = time_sampler.get_time_steps(indices)
        assert computed_time_steps.shape == shape

        assert torch.all(computed_time_steps > 0.0)
        assert torch.all(computed_time_steps <= 1.0)

        # The random values should come from an array of the form [dt, 2 dt, 3 dt,.... N dt]
        delta_time = 1. / (total_time_steps - 1.)
        number_of_time_segments = torch.round(computed_time_steps / delta_time)
        expected_time_steps = number_of_time_segments * delta_time

        assert torch.all(torch.isclose(expected_time_steps, computed_time_steps))

    @pytest.fixture()
    def all_indices_and_times(self, total_time_steps):
        dt = torch.tensor(1. / (total_time_steps - 1.)).to(torch.double)
        all_time_indices = torch.arange(total_time_steps)
        all_times = (all_time_indices * dt).float()
        return all_time_indices, all_times

    def test_forward_iterator(self, all_indices_and_times, time_sampler):

        all_time_indices, all_times = all_indices_and_times
        expected_indices = all_time_indices[:-1]
        expected_times = all_times[:-1]

        computed_indices = []
        computed_times = []
        for index, time in time_sampler.get_forward_iterator():
            computed_indices.append(index)
            computed_times.append(time)

        computed_indices = torch.tensor(computed_indices)
        computed_times = torch.tensor(computed_times)

        assert torch.all(torch.isclose(computed_times, expected_times))
        assert torch.all(torch.isclose(computed_indices, expected_indices))

    def test_backward_iterator(self, all_indices_and_times, time_sampler):
        all_time_indices, all_times = all_indices_and_times
        expected_indices = all_time_indices[1:].flip(dims=(0,))
        expected_times = all_times[1:].flip(dims=(0,))

        computed_indices = []
        computed_times = []
        for index, time in time_sampler.get_backward_iterator():
            computed_indices.append(index)
            computed_times.append(time)

        computed_indices = torch.tensor(computed_indices)
        computed_times = torch.tensor(computed_times)

        assert torch.all(torch.isclose(computed_times, expected_times))
        assert torch.all(torch.isclose(computed_indices, expected_indices))

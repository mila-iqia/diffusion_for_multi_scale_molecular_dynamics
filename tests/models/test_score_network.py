import pytest
import torch

from crystal_diffusion.models.score_network import (BaseScoreNetworkParameters,
                                                    MLPScoreNetwork,
                                                    MLPScoreNetworkParameters,
                                                    ScoreNetwork)


@pytest.mark.parametrize("spatial_dimension", [2, 3])
class TestScoreNetworkCheck:

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(123)

    @pytest.fixture()
    def base_score_network(self, spatial_dimension):
        return ScoreNetwork(BaseScoreNetworkParameters(spatial_dimension=spatial_dimension))

    @pytest.fixture()
    def good_batch(self, spatial_dimension):
        batch_size = 16
        positions = torch.rand(batch_size, 8, spatial_dimension)
        times = torch.rand(batch_size, 1)
        return {ScoreNetwork.position_key: positions, ScoreNetwork.timestep_key: times}

    @pytest.fixture()
    def bad_batch(self, good_batch, problem):

        bad_batch = dict(good_batch)

        match problem:
            case "position_name":
                bad_batch['bad_position_name'] = bad_batch[ScoreNetwork.position_key]
                del bad_batch[ScoreNetwork.position_key]

            case "position_shape":
                shape = bad_batch[ScoreNetwork.position_key].shape
                bad_batch[ScoreNetwork.position_key] = \
                    bad_batch[ScoreNetwork.position_key].reshape(shape[0], shape[1] // 2, shape[2] * 2)

            case "position_range1":
                bad_batch[ScoreNetwork.position_key][0, 0, 0] = 1.01

            case "position_range2":
                bad_batch[ScoreNetwork.position_key][1, 0, 0] = -0.01

            case "time_name":
                bad_batch['bad_time_name'] = bad_batch[ScoreNetwork.timestep_key]
                del bad_batch[ScoreNetwork.timestep_key]

            case "time_shape":
                shape = bad_batch['time'].shape
                bad_batch[ScoreNetwork.timestep_key] = (
                    bad_batch[ScoreNetwork.timestep_key].reshape(shape[0] // 2, shape[1] * 2))

            case "time_range1":
                bad_batch[ScoreNetwork.timestep_key][5, 0] = 2.00
            case "time_range2":
                bad_batch[ScoreNetwork.timestep_key][0, 0] = -0.05

        return bad_batch

    def test_check_batch_good(self, base_score_network, good_batch):
        base_score_network._check_batch(good_batch)

    @pytest.mark.parametrize('problem', ['position_name', 'time_name', 'position_shape',
                                         'time_shape', 'position_range1', 'position_range2',
                                         'time_range1', 'time_range2'])
    def test_check_batch_bad(self, base_score_network, bad_batch):
        with pytest.raises(AssertionError):
            base_score_network._check_batch(bad_batch)


@pytest.mark.parametrize("spatial_dimension", [2, 3])
@pytest.mark.parametrize("n_hidden_dimensions", [1, 2, 3])
@pytest.mark.parametrize("hidden_dimensions_size", [8, 16])
class TestMLPScoreNetwork:

    @pytest.fixture()
    def batch_size(self):
        return 16

    @pytest.fixture()
    def number_of_atoms(self):
        return 8

    @pytest.fixture()
    def expected_score_shape(self, batch_size, number_of_atoms, spatial_dimension):
        return batch_size, number_of_atoms, spatial_dimension

    @pytest.fixture()
    def good_batch(self, batch_size, number_of_atoms, spatial_dimension):
        positions = torch.rand(batch_size, number_of_atoms, spatial_dimension)
        times = torch.rand(batch_size, 1)
        return {ScoreNetwork.position_key: positions, ScoreNetwork.timestep_key: times}

    @pytest.fixture()
    def bad_batch(self, batch_size, number_of_atoms, spatial_dimension):
        positions = torch.rand(batch_size, number_of_atoms // 2, spatial_dimension)
        times = torch.rand(batch_size, 1)
        return {ScoreNetwork.position_key: positions, ScoreNetwork.timestep_key: times}

    @pytest.fixture()
    def score_network(self, number_of_atoms, spatial_dimension, n_hidden_dimensions, hidden_dimensions_size):
        hyper_params = MLPScoreNetworkParameters(spatial_dimension=spatial_dimension,
                                                 number_of_atoms=number_of_atoms,
                                                 n_hidden_dimensions=n_hidden_dimensions,
                                                 hidden_dimensions_size=hidden_dimensions_size)
        return MLPScoreNetwork(hyper_params)

    def test_check_batch_bad(self, score_network, bad_batch):
        with pytest.raises(AssertionError):
            score_network._check_batch(bad_batch)

    def test_check_batch_good(self, score_network, good_batch):
        score_network._check_batch(good_batch)

    def test_output_shape(self, score_network, good_batch, expected_score_shape):
        scores = score_network(good_batch)
        assert scores.shape == expected_score_shape
